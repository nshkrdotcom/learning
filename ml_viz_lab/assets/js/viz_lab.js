import * as THREE from "three"
import dagre from "dagre"
import {OrbitControls} from "three/examples/jsm/controls/OrbitControls.js"
import {EditorState, StateEffect, StateField} from "@codemirror/state"
import {EditorView, Decoration, lineNumbers} from "@codemirror/view"
import {syntaxHighlighting, defaultHighlightStyle} from "@codemirror/language"
import {elixir} from "codemirror-lang-elixir"

const activeLineEffect = StateEffect.define()

const activeLineField = StateField.define({
  create() {
    return Decoration.none
  },
  update(decorations, tr) {
    decorations = decorations.map(tr.changes)
    for (const effect of tr.effects) {
      if (effect.is(activeLineEffect)) {
        const lineNumber = effect.value
        if (!lineNumber) return Decoration.none
        const line = tr.state.doc.line(Math.min(lineNumber, tr.state.doc.lines))
        decorations = Decoration.set([
          Decoration.line({class: "cm-viz-active-line"}).range(line.from),
        ])
      }
    }
    return decorations
  },
  provide: field => EditorView.decorations.from(field),
})

const VizLabHook = {
  mounted() {
    this.trace = JSON.parse(this.el.dataset.trace)
    this.lessons = JSON.parse(this.el.dataset.lessons)
    this.subjects = JSON.parse(this.el.dataset.subjects)
    this.step = Number(new URLSearchParams(window.location.search).get("step") || 0)
    this.step = clamp(this.step, 0, this.trace.events.length - 1)
    this.speed = 1
    this.timer = null
    this.codeMode = "lesson"
    this.viewMode = "execution"
    this.showLabels = true
    this.explanationLevel = "intuition"
    this.activeFile = null
    this.manualSourceFile = null
    this.selected = null

    this.refs = {
      codeTitle: this.el.querySelector("#code-title"),
      sourceTabs: this.el.querySelector("#source-tabs"),
      codeEditor: this.el.querySelector("#code-editor"),
      variableInspector: this.el.querySelector("#variable-inspector"),
      graphTitle: this.el.querySelector("#graph-title"),
      phaseLabel: this.el.querySelector("#phase-label"),
      canvas: this.el.querySelector("#graph-canvas"),
      minimap: this.el.querySelector("#graph-minimap"),
      selectionCard: this.el.querySelector("#selection-card"),
      lessonTitle: this.el.querySelector("#lesson-title"),
      programSelector: this.el.querySelector("#program-selector"),
      progressStrip: this.el.querySelector("#progress-strip"),
      lessonCard: this.el.querySelector("#lesson-card"),
      conceptDrawer: this.el.querySelector("#concept-drawer"),
      checkpointRow: this.el.querySelector("#checkpoint-row"),
      scrubber: this.el.querySelector("#timeline-scrubber"),
      stepCounter: this.el.querySelector("#step-counter"),
      speedSelect: this.el.querySelector("#speed-select"),
    }

    this.sourceById = new Map(this.trace.sources.map(source => [source.id, source]))
    this.eventBySourceLine = new Map()
    for (const event of this.trace.events) {
      for (const source of [event.source, event.implementation_source]) {
        const key = `${source.file}:${source.line}`
        if (!this.eventBySourceLine.has(key)) this.eventBySourceLine.set(key, event.index)
      }
    }

    this.initCode()
    this.initGraph()
    this.bindControls()
    this.renderAll(false)
  },

  destroyed() {
    this.stop()
    if (this.editor) this.editor.destroy()
    if (this.renderer) this.renderer.dispose()
    window.removeEventListener("resize", this.resizeHandler)
    window.removeEventListener("keydown", this.keyHandler)
  },

  initCode() {
    this.renderSourceTabs()
  },

  initGraph() {
    const canvas = this.refs.canvas
    const wrap = canvas.parentElement
    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color("#0e0e12")
    this.camera = new THREE.PerspectiveCamera(45, 1, 1, 5000)
    this.camera.position.set(0, 0, 900)
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: true})
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableRotate = false
    this.controls.enableDamping = true
    this.controls.zoomSpeed = 0.8
    this.raycaster = new THREE.Raycaster()
    this.raycaster.params.Line.threshold = 12
    this.pointer = new THREE.Vector2()
    this.nodeObjects = new Map()
    this.edgeObjects = new Map()
    this.activePulse = 0

    const ambient = new THREE.AmbientLight("#ffffff", 1.8)
    this.scene.add(ambient)

    const light = new THREE.DirectionalLight("#ffffff", 2.0)
    light.position.set(0, 0, 600)
    this.scene.add(light)

    this.graphGroup = new THREE.Group()
    this.scene.add(this.graphGroup)
    this.layoutGraph()
    this.createGraphObjects()

    canvas.addEventListener("pointerdown", event => this.handleCanvasPointer(event))

    this.resizeHandler = () => this.resizeGraph()
    window.addEventListener("resize", this.resizeHandler)
    this.resizeGraph()

    const animate = () => {
      this.activePulse += 0.035
      for (const object of this.nodeObjects.values()) {
        if (object.isActive) {
          const scale = 1 + Math.sin(this.activePulse) * 0.04
          object.group.scale.set(scale, scale, scale)
        }
      }
      this.controls.update()
      this.renderer.render(this.scene, this.camera)
      this.animationFrame = requestAnimationFrame(animate)
    }
    animate()
  },

  layoutGraph() {
    const graph = new dagre.graphlib.Graph()
    graph.setGraph({rankdir: "LR", nodesep: 54, ranksep: 120, marginx: 40, marginy: 40})
    graph.setDefaultEdgeLabel(() => ({}))

    for (const node of this.trace.final_graph.nodes) {
      graph.setNode(String(node.id), {width: 96, height: 58})
    }

    for (const edge of this.trace.final_graph.edges) {
      graph.setEdge(String(edge.from), String(edge.to))
    }

    dagre.layout(graph)
    this.layout = new Map()

    for (const node of this.trace.final_graph.nodes) {
      const pos = graph.node(String(node.id)) || {x: 0, y: 0}
      this.layout.set(String(node.id), {x: pos.x - graph.graph().width / 2, y: -(pos.y - graph.graph().height / 2)})
    }
  },

  createGraphObjects() {
    for (const node of this.trace.final_graph.nodes) {
      const id = String(node.id)
      const pos = this.layout.get(id) || {x: 0, y: 0}
      const group = new THREE.Group()
      group.position.set(pos.x, pos.y, 0)

      const geometry = new THREE.BoxGeometry(node.kind === "leaf" ? 88 : 78, node.kind === "leaf" ? 48 : 38, 16)
      const material = new THREE.MeshStandardMaterial({
        color: colorForNode(node),
        emissive: "#000000",
        roughness: 0.45,
        metalness: 0.08,
      })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.userData = {type: "node", id, node}
      group.add(mesh)

      const label = makeTextSprite(labelText(node, null), 240, 82)
      label.position.set(0, 0, 14)
      group.add(label)

      const outline = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        new THREE.LineBasicMaterial({color: node.is_output ? "#ffffff" : "#2b2d3a", transparent: true, opacity: node.is_output ? 0.95 : 0.45})
      )
      outline.position.copy(mesh.position)
      group.add(outline)

      this.graphGroup.add(group)
      this.nodeObjects.set(id, {group, mesh, label, material, node, isActive: false})
    }

    for (const edge of this.trace.final_graph.edges) {
      const from = this.layout.get(String(edge.from))
      const to = this.layout.get(String(edge.to))
      if (!from || !to) continue

      const points = [
        new THREE.Vector3(from.x + 44, from.y, -7),
        new THREE.Vector3((from.x + to.x) / 2, (from.y + to.y) / 2, -7),
        new THREE.Vector3(to.x - 44, to.y, -7),
      ]
      const curve = new THREE.CatmullRomCurve3(points)
      const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(18))
      const material = new THREE.LineBasicMaterial({color: "#44485f", transparent: true, opacity: 0.55})
      const line = new THREE.Line(geometry, material)
      line.userData = {type: "edge", id: edge.id, edge}
      this.graphGroup.add(line)

      const label = makeTextSprite(edge.label || "", 160, 42, 18)
      label.position.set((from.x + to.x) / 2, (from.y + to.y) / 2 + 16, 2)
      this.graphGroup.add(label)

      this.edgeObjects.set(edge.id, {line, label, material, edge})
    }
  },

  bindControls() {
    this.el.querySelectorAll("[data-action]").forEach(button => {
      button.addEventListener("click", () => this.handleAction(button.dataset.action))
    })

    this.refs.speedSelect.addEventListener("change", event => {
      this.speed = Number(event.target.value)
      if (this.timer) {
        this.stop()
        this.play()
      }
    })

    this.refs.scrubber.max = Math.max(this.trace.events.length - 1, 0)
    this.refs.scrubber.addEventListener("input", event => this.setStep(Number(event.target.value), false))

    this.el.querySelectorAll("[data-code-mode]").forEach(button => {
      button.addEventListener("click", () => {
        this.codeMode = button.dataset.codeMode
        this.el.querySelectorAll("[data-code-mode]").forEach(btn => btn.classList.toggle("active", btn === button))
        this.renderCode()
      })
    })

    this.el.querySelectorAll("[data-view-mode]").forEach(button => {
      button.addEventListener("click", () => {
        this.viewMode = button.dataset.viewMode
        this.el.querySelectorAll("[data-view-mode]").forEach(btn => btn.classList.toggle("active", btn === button))
        this.renderGraph()
      })
    })

    this.el.querySelector("[data-toggle-labels]").addEventListener("click", event => {
      this.showLabels = !this.showLabels
      event.currentTarget.classList.toggle("active", this.showLabels)
      for (const object of this.nodeObjects.values()) object.label.visible = this.showLabels
      for (const object of this.edgeObjects.values()) object.label.visible = this.showLabels
    })

    this.el.querySelector("[data-reset-camera]").addEventListener("click", () => this.resetCamera())

    this.keyHandler = event => {
      if (event.target.matches("input, textarea, select")) return
      if (event.key === " ") {
        event.preventDefault()
        this.toggle()
      } else if (event.key === "ArrowRight") {
        event.preventDefault()
        this.setStep(this.step + (event.shiftKey ? 10 : 1), true)
      } else if (event.key === "ArrowLeft") {
        event.preventDefault()
        this.setStep(this.step - (event.shiftKey ? 10 : 1), true)
      } else if (event.key === "Home") {
        event.preventDefault()
        this.setStep(0, false)
      } else if (event.key === "End") {
        event.preventDefault()
        this.setStep(this.trace.events.length - 1, false)
      } else if (/^[1-9]$/.test(event.key)) {
        const checkpoint = this.trace.checkpoints[Number(event.key) - 1]
        if (checkpoint) this.setStep(checkpoint.step, false)
      }
    }
    window.addEventListener("keydown", this.keyHandler)
  },

  handleAction(action) {
    switch (action) {
      case "start": return this.setStep(0, false)
      case "back10": return this.setStep(this.step - 10, false)
      case "back1": return this.setStep(this.step - 1, true)
      case "toggle": return this.toggle()
      case "forward1": return this.setStep(this.step + 1, true)
      case "forward10": return this.setStep(this.step + 10, false)
      case "end": return this.setStep(this.trace.events.length - 1, false)
    }
  },

  toggle() {
    if (this.timer) this.stop()
    else this.play()
  },

  play() {
    this.manualSourceFile = null
    this.el.querySelector("[data-action='toggle']").textContent = "Pause"
    const tick = () => {
      if (this.step >= this.trace.events.length - 1) {
        this.stop()
        return
      }
      this.setStep(this.step + 1, true, false)
      this.timer = window.setTimeout(tick, Math.max(60, 600 / this.speed))
    }
    this.timer = window.setTimeout(tick, Math.max(60, 600 / this.speed))
  },

  stop() {
    if (this.timer) window.clearTimeout(this.timer)
    this.timer = null
    const toggle = this.el.querySelector("[data-action='toggle']")
    if (toggle) toggle.textContent = "Play"
  },

  setStep(nextStep, animated = false, updateUrl = true) {
    this.step = clamp(nextStep, 0, this.trace.events.length - 1)
    this.renderAll(animated)
    if (updateUrl) this.updateUrl()
  },

  renderAll(animated) {
    this.renderControls()
    this.renderCode()
    this.renderGraph(animated)
    this.renderTeaching()
  },

  renderControls() {
    this.refs.scrubber.value = this.step
    this.refs.stepCounter.textContent = `Step ${this.step + 1} / ${this.trace.events.length}`
    this.refs.checkpointRow.innerHTML = this.trace.checkpoints.map((checkpoint, index) => {
      const active = checkpoint.step <= this.step ? "active" : ""
      return `<button type="button" class="checkpoint ${active}" data-step="${checkpoint.step}" title="${index + 1}">${escapeHtml(checkpoint.label)}</button>`
    }).join("")
    this.refs.checkpointRow.querySelectorAll("button").forEach(button => {
      button.addEventListener("click", () => this.setStep(Number(button.dataset.step), false))
    })
  },

  renderCode() {
    const event = this.currentEvent()
    let sourceRef = this.codeMode === "lesson" ? event.source : event.implementation_source
    if (this.manualSourceFile) {
      sourceRef = {
        ...sourceRef,
        file: this.manualSourceFile,
        line: this.manualSourceFile === sourceRef.file ? sourceRef.line : 1,
      }
    }
    const source = this.sourceById.get(sourceRef.file) || this.trace.sources[0]

    if (this.activeFile !== source.id || !this.editor) {
      this.activeFile = source.id
      if (this.editor) this.editor.destroy()
      this.editor = new EditorView({
        state: EditorState.create({
          doc: source.source,
          extensions: [
            lineNumbers(),
            elixir(),
            syntaxHighlighting(defaultHighlightStyle),
            EditorState.readOnly.of(true),
            EditorView.editable.of(false),
            activeLineField,
            EditorView.theme({
              "&": {height: "100%"},
              ".cm-scroller": {fontFamily: "'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, monospace"},
            }),
          ],
        }),
        parent: this.refs.codeEditor,
      })

      this.editor.dom.addEventListener("click", event => this.handleEditorClick(event))
      this.renderSourceTabs()
    }

    this.refs.codeTitle.textContent = source.title
    this.editor.dispatch({effects: activeLineEffect.of(sourceRef.line)})
    const line = this.editor.state.doc.line(Math.min(sourceRef.line, this.editor.state.doc.lines))
    this.editor.dispatch({selection: {anchor: line.from}, scrollIntoView: true})
    this.renderVariableInspector()
  },

  renderSourceTabs() {
    this.refs.sourceTabs.innerHTML = this.trace.sources.map(source => {
      const active = source.id === this.activeFile ? "active" : ""
      return `<button type="button" class="source-tab ${active}" data-file="${source.id}">${escapeHtml(source.title)}</button>`
    }).join("")
    this.refs.sourceTabs.querySelectorAll("button").forEach(button => {
      button.addEventListener("click", () => {
        this.activeFile = null
        this.manualSourceFile = button.dataset.file
        this.codeMode = button.dataset.file === "lesson.ex" ? "lesson" : "source"
        this.renderCode()
      })
    })
  },

  handleEditorClick(event) {
    const pos = this.editor.posAtCoords({x: event.clientX, y: event.clientY})
    if (pos == null) return
    const line = this.editor.state.doc.lineAt(pos).number
    const key = `${this.activeFile}:${line}`
    const step = this.eventBySourceLine.get(key)
    if (step != null) this.setStep(step, false)
  },

  renderVariableInspector() {
    const event = this.currentEvent()
    const visible = new Set((event.snapshot.visible_nodes || []).map(String))
    const gradients = event.snapshot.gradients || {}

    const rows = this.trace.final_graph.nodes
      .filter(node => visible.has(String(node.id)))
      .map(node => {
        const grad = gradients[String(node.id)] ?? gradients[node.id]
        return `
          <button type="button" class="var-row" data-node="${node.id}">
            <span>${escapeHtml(node.title || node.display_id)}</span>
            <b>${formatNumber(node.data)}</b>
            <em>${grad == null ? "-" : formatNumber(grad)}</em>
          </button>
        `
      })
      .join("")

    this.refs.variableInspector.innerHTML = `<div class="inspector-title">Values</div>${rows}`
    this.refs.variableInspector.querySelectorAll("[data-node]").forEach(button => {
      button.addEventListener("click", () => this.selectNode(String(button.dataset.node)))
    })
  },

  renderGraph() {
    const event = this.currentEvent()
    const visible = new Set((event.snapshot.visible_nodes || []).map(String))
    const gradients = event.snapshot.gradients || {}
    const structural = this.viewMode === "structural"

    this.refs.phaseLabel.textContent = titleCase(event.phase)
    this.refs.graphTitle.textContent = event.title

    for (const [id, object] of this.nodeObjects) {
      const isVisible = structural || visible.has(id)
      const isActive = String(event.snapshot.active_node) === id
      const grad = gradients[id] ?? gradients[Number(id)]
      object.group.visible = isVisible
      object.isActive = isActive
      object.group.scale.set(1, 1, 1)
      object.material.color.set(colorForNode(object.node))
      object.material.emissive.set(isActive ? "#6b3d00" : "#000000")
      updateTextSprite(object.label, labelText(object.node, grad), 240, 82)
    }

    for (const [id, object] of this.edgeObjects) {
      const fromVisible = visible.has(String(object.edge.from))
      const toVisible = visible.has(String(object.edge.to))
      const isVisible = structural || (fromVisible && toVisible)
      const isActive = event.snapshot.active_edge === id
      object.line.visible = isVisible
      object.label.visible = isVisible && this.showLabels
      object.material.color.set(isActive ? "#16a34a" : "#44485f")
      object.material.opacity = isActive ? 1 : 0.55
    }

    this.renderMinimap(visible, structural)
    this.renderSelection()
  },

  renderMinimap(visible, structural) {
    const positions = Array.from(this.layout.values())
    const xs = positions.map(p => p.x)
    const ys = positions.map(p => p.y)
    const minX = Math.min(...xs, -1)
    const maxX = Math.max(...xs, 1)
    const minY = Math.min(...ys, -1)
    const maxY = Math.max(...ys, 1)

    this.refs.minimap.innerHTML = this.trace.final_graph.nodes.map(node => {
      const id = String(node.id)
      const pos = this.layout.get(id) || {x: 0, y: 0}
      const left = ((pos.x - minX) / Math.max(maxX - minX, 1)) * 86 + 7
      const top = ((maxY - pos.y) / Math.max(maxY - minY, 1)) * 70 + 10
      const active = String(this.currentEvent().snapshot.active_node) === id ? "active" : ""
      const hidden = !structural && !visible.has(id) ? "hidden-dot" : ""
      return `<span class="mini-dot ${active} ${hidden}" style="left:${left}%;top:${top}%"></span>`
    }).join("")
  },

  renderTeaching() {
    const event = this.currentEvent()
    this.refs.lessonTitle.textContent = this.trace.title
    this.renderProgramSelector()
    this.renderProgress()

    const teaching = event.teaching || {}
    const levels = ["intuition", "mechanism", "math", "elixir"]
    const current = teaching[this.explanationLevel] || teaching.intuition || ""

    this.refs.lessonCard.innerHTML = `
      <div class="lesson-card-header">
        <span>${escapeHtml(titleCase(event.phase))}</span>
        <strong>${escapeHtml(event.title)}</strong>
      </div>
      <div class="depth-tabs">
        ${levels.map(level => `<button type="button" class="${level === this.explanationLevel ? "active" : ""}" data-level="${level}">${escapeHtml(titleCase(level))}</button>`).join("")}
      </div>
      <p class="lesson-copy">${escapeHtml(current)}</p>
      ${this.renderEventFact(event)}
      <div class="concept-list">
        ${(event.concepts || []).map(id => {
          const concept = this.trace.concepts.find(item => item.id === id)
          return concept ? `<button type="button" data-concept="${concept.id}">${escapeHtml(concept.title)}</button>` : ""
        }).join("")}
      </div>
    `

    this.refs.lessonCard.querySelectorAll("[data-level]").forEach(button => {
      button.addEventListener("click", () => {
        this.explanationLevel = button.dataset.level
        this.renderTeaching()
      })
    })

    this.refs.lessonCard.querySelectorAll("[data-concept]").forEach(button => {
      button.addEventListener("click", () => this.openConcept(button.dataset.concept))
    })
  },

  renderEventFact(event) {
    if (event.gradient) {
      const gradient = event.gradient
      return `
        <dl class="fact-grid">
          <div><dt>upstream</dt><dd>${formatNumber(gradient.upstream ?? gradient.contribution)}</dd></div>
          <div><dt>local</dt><dd>${formatNumber(gradient.local_gradient ?? 1)}</dd></div>
          <div><dt>after</dt><dd>${formatNumber(gradient.after)}</dd></div>
        </dl>
      `
    }

    if (event.parameter_update) {
      const update = event.parameter_update
      return `
        <dl class="fact-grid">
          <div><dt>old</dt><dd>${formatNumber(update.old_data)}</dd></div>
          <div><dt>grad</dt><dd>${formatNumber(update.gradient)}</dd></div>
          <div><dt>new</dt><dd>${formatNumber(update.new_data)}</dd></div>
        </dl>
      `
    }

    if (event.metrics) {
      return `
        <dl class="fact-grid">
          <div><dt>loss</dt><dd>${formatNumber(event.metrics.loss)}</dd></div>
          <div><dt>w</dt><dd>${formatNumber(event.metrics.weight)}</dd></div>
          <div><dt>b</dt><dd>${formatNumber(event.metrics.bias)}</dd></div>
        </dl>
      `
    }

    if (event.value) {
      return `
        <dl class="fact-grid">
          <div><dt>node</dt><dd>${escapeHtml(event.value.display_id)}</dd></div>
          <div><dt>op</dt><dd>${escapeHtml(event.value.op)}</dd></div>
          <div><dt>data</dt><dd>${formatNumber(event.value.data)}</dd></div>
        </dl>
      `
    }

    return ""
  },

  renderProgramSelector() {
    const current = this.trace.lesson_id
    const groups = new Map()
    for (const lesson of this.lessons) {
      if (!groups.has(lesson.level)) groups.set(lesson.level, [])
      groups.get(lesson.level).push(lesson)
    }

    this.refs.programSelector.innerHTML = Array.from(groups.entries()).map(([level, lessons]) => `
      <section>
        <h3>${escapeHtml(level)}</h3>
        ${lessons.map(lesson => `
          <button type="button" class="program-card ${lesson.id === current ? "active" : ""}" data-lesson="${lesson.id}">
            <strong>${escapeHtml(lesson.title)}</strong>
            <span>${escapeHtml(lesson.description)}</span>
          </button>
        `).join("")}
      </section>
    `).join("")

    this.refs.programSelector.querySelectorAll("[data-lesson]").forEach(button => {
      button.addEventListener("click", () => {
        const params = new URLSearchParams(window.location.search)
        params.set("lesson", button.dataset.lesson)
        params.delete("step")
        window.location.search = params.toString()
      })
    })
  },

  renderProgress() {
    const total = Math.max(this.trace.events.length - 1, 1)
    const percent = (this.step / total) * 100
    this.refs.progressStrip.innerHTML = `
      <div class="progress-track"><span style="width:${percent}%"></span></div>
      <div class="progress-meta"><span>${escapeHtml(this.currentEvent().phase)}</span><span>${this.step + 1}/${this.trace.events.length}</span></div>
    `
  },

  openConcept(id) {
    const concept = this.trace.concepts.find(item => item.id === id)
    if (!concept) return
    this.refs.conceptDrawer.innerHTML = `
      <article>
        <button type="button" class="drawer-close" title="Close">x</button>
        <h3>${escapeHtml(concept.title)}</h3>
        <p>${escapeHtml(concept.body)}</p>
      </article>
    `
    this.refs.conceptDrawer.querySelector("button").addEventListener("click", () => {
      this.refs.conceptDrawer.innerHTML = ""
    })
  },

  handleCanvasPointer(event) {
    const rect = this.refs.canvas.getBoundingClientRect()
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
    this.raycaster.setFromCamera(this.pointer, this.camera)
    const objects = [
      ...Array.from(this.nodeObjects.values()).map(object => object.mesh),
      ...Array.from(this.edgeObjects.values()).map(object => object.line),
    ].filter(object => object.visible)

    const hit = this.raycaster.intersectObjects(objects, false)[0]
    if (!hit) return

    if (hit.object.userData.type === "node") this.selectNode(String(hit.object.userData.id))
    if (hit.object.userData.type === "edge") this.selectEdge(hit.object.userData.id)
  },

  selectNode(id) {
    const object = this.nodeObjects.get(String(id))
    if (!object) return
    this.selected = {type: "node", id: String(id), node: object.node}
    this.renderSelection()
  },

  selectEdge(id) {
    const object = this.edgeObjects.get(id)
    if (!object) return
    this.selected = {type: "edge", id, edge: object.edge}
    this.renderSelection()
  },

  renderSelection() {
    if (!this.selected) {
      this.refs.selectionCard.classList.add("is-empty")
      this.refs.selectionCard.innerHTML = ""
      return
    }

    this.refs.selectionCard.classList.remove("is-empty")
    const gradients = this.currentEvent().snapshot.gradients || {}

    if (this.selected.type === "node") {
      const node = this.selected.node
      const grad = gradients[String(node.id)] ?? gradients[node.id]
      const edges = this.trace.final_graph.edges.filter(edge => String(edge.from) === String(node.id) || String(edge.to) === String(node.id))

      this.refs.selectionCard.innerHTML = `
        <article>
          <button type="button" class="drawer-close" title="Close">x</button>
          <h3>${escapeHtml(node.title)}</h3>
          <dl>
            <div><dt>op</dt><dd>${escapeHtml(node.op)}</dd></div>
            <div><dt>data</dt><dd>${formatNumber(node.data)}</dd></div>
            <div><dt>grad</dt><dd>${grad == null ? "-" : formatNumber(grad)}</dd></div>
          </dl>
          <div class="edge-list">
            ${edges.map(edge => `<button type="button" data-edge="${edge.id}">${escapeHtml(edge.label || edge.id)}</button>`).join("")}
          </div>
        </article>
      `
      this.refs.selectionCard.querySelector(".drawer-close").addEventListener("click", () => {
        this.selected = null
        this.renderSelection()
      })
      this.refs.selectionCard.querySelectorAll("[data-edge]").forEach(button => {
        button.addEventListener("click", () => this.selectEdge(button.dataset.edge))
      })
    } else {
      const edge = this.selected.edge
      this.refs.selectionCard.innerHTML = `
        <article>
          <button type="button" class="drawer-close" title="Close">x</button>
          <h3>Edge</h3>
          <dl>
            <div><dt>from</dt><dd>${escapeHtml(String(edge.from))}</dd></div>
            <div><dt>to</dt><dd>${escapeHtml(String(edge.to))}</dd></div>
            <div><dt>local</dt><dd>${edge.local_gradient == null ? "-" : formatNumber(edge.local_gradient)}</dd></div>
          </dl>
        </article>
      `
      this.refs.selectionCard.querySelector(".drawer-close").addEventListener("click", () => {
        this.selected = null
        this.renderSelection()
      })
    }
  },

  resizeGraph() {
    const wrap = this.refs.canvas.parentElement
    const width = Math.max(wrap.clientWidth, 300)
    const height = Math.max(wrap.clientHeight, 260)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(width, height, false)
  },

  resetCamera() {
    this.camera.position.set(0, 0, 900)
    this.controls.target.set(0, 0, 0)
    this.controls.update()
  },

  updateUrl() {
    const params = new URLSearchParams(window.location.search)
    params.set("lesson", this.trace.lesson_id)
    params.set("step", String(this.step))
    history.replaceState(null, "", `${window.location.pathname}?${params}`)
  },

  currentEvent() {
    return this.trace.events[this.step]
  },
}

function colorForNode(node) {
  if (node.is_output) return "#f5a623"
  if (node.kind === "leaf") return "#4b50c8"
  if (node.op === "+") return "#c8930a"
  if (node.op === "*") return "#c8550a"
  if (node.op.startsWith("^")) return "#b6462d"
  if (node.op === "relu") return "#16a34a"
  if (node.op === "loss") return "#f5a623"
  if (node.op === "parameter") return "#4b50c8"
  return "#64748b"
}

function labelText(node, grad) {
  const title = node.title || node.display_id
  const gradText = grad == null ? "-" : formatNumber(grad)
  return `${title}\n${node.op}  data ${formatNumber(node.data)}\ngrad ${gradText}`
}

function makeTextSprite(text, width = 220, height = 72, fontSize = 20) {
  const texture = makeTextTexture(text, width, height, fontSize)
  const material = new THREE.SpriteMaterial({map: texture, transparent: true, depthTest: false})
  const sprite = new THREE.Sprite(material)
  sprite.scale.set(width / 2.5, height / 2.5, 1)
  return sprite
}

function updateTextSprite(sprite, text, width = 220, height = 72, fontSize = 20) {
  if (!sprite.visible) return
  if (sprite.userData.text === text) return
  sprite.userData.text = text
  const old = sprite.material.map
  sprite.material.map = makeTextTexture(text, width, height, fontSize)
  sprite.material.needsUpdate = true
  if (old) old.dispose()
}

function makeTextTexture(text, width, height, fontSize) {
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext("2d")
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = "rgba(10, 10, 16, 0.78)"
  roundRect(ctx, 0, 0, width, height, 10)
  ctx.fill()
  ctx.fillStyle = "#f5f5fb"
  ctx.font = `${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`
  ctx.textAlign = "center"
  ctx.textBaseline = "middle"
  const lines = String(text).split("\n")
  const start = height / 2 - ((lines.length - 1) * fontSize) / 2
  lines.forEach((line, index) => ctx.fillText(line.slice(0, 28), width / 2, start + index * fontSize))
  const texture = new THREE.CanvasTexture(canvas)
  texture.needsUpdate = true
  return texture
}

function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath()
  ctx.moveTo(x + radius, y)
  ctx.arcTo(x + width, y, x + width, y + height, radius)
  ctx.arcTo(x + width, y + height, x, y + height, radius)
  ctx.arcTo(x, y + height, x, y, radius)
  ctx.arcTo(x, y, x + width, y, radius)
  ctx.closePath()
}

function formatNumber(value) {
  if (value == null || Number.isNaN(Number(value))) return "-"
  const number = Number(value)
  if (Math.abs(number) >= 1000 || Math.abs(number) < 0.001 && number !== 0) return number.toExponential(2)
  return number.toFixed(4).replace(/\.?0+$/, "")
}

function titleCase(value) {
  return String(value || "").replace(/_/g, " ").replace(/\b\w/g, letter => letter.toUpperCase())
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max)
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;")
}

export {VizLabHook}
