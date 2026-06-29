import * as THREE from "three"
import dagre from "dagre"
import {OrbitControls} from "three/examples/jsm/controls/OrbitControls.js"
import {EditorState, StateEffect, StateField} from "@codemirror/state"
import {EditorView, Decoration, lineNumbers} from "@codemirror/view"
import {syntaxHighlighting, defaultHighlightStyle} from "@codemirror/language"
import {elixir} from "codemirror-lang-elixir"
import {clamp, escapeHtml, formatNumber, titleCase} from "./viz/format"
import {initialPlayback, nextLoopStep, playbackReducer} from "./viz/playback"
import {readStepFromUrl, writeStepToUrl} from "./viz/url_state"
import {buildSourceLineIndex, parsePayload} from "./viz/trace_store"
import {
  bindingRows,
  canSendLiveAction,
  initialLiveState,
  normalizeSource,
  reduceExecutionEvent,
} from "./viz/live_state"

function liveDebug(...args) {
  if (window.VIZ_LIVE_DEBUG === true) {
    console.debug("[ml-viz-live]", ...args)
  }
}

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
    this.trace = null
    this.lessons = parsePayload(this.el.dataset.lessons, [])
    this.subjects = parsePayload(this.el.dataset.subjects, [])
    this.executionMode = this.el.dataset.executionMode || "replay"
    this.currentSubjectId = new URLSearchParams(window.location.search).get("subject") || this.subjects[0]?.id || "micrograd"
    this.currentLessonId = new URLSearchParams(window.location.search).get("lesson") || this.lessons[0]?.id || "x_squared"
    this.live = initialLiveState(parsePayload(this.el.dataset.liveSource, null))
    this.playback = initialPlayback(0, 0)
    this.step = 0
    this.speed = 1
    this.timer = null
    this.codeMode = "lesson"
    this.code = {
      activeSourceId: null,
      activeLineStart: null,
      activeLineEnd: null,
      lastKnownGoodSource: normalizeSource(parsePayload(this.el.dataset.liveSource, null)),
      error: null,
    }
    this.viewMode = "execution"
    this.showLabels = true
    this.showLocalDerivatives = true
    this.showGradients = true
    this.explanationLevel = "intuition"
    this.activeFile = null
    this.manualSourceFile = null
    this.selected = null
    this.compareStep = null
    this.controlsBound = false

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
      executionStatusCard: this.el.querySelector("#execution-status-card"),
      liveStatus: this.el.querySelector("#live-status"),
      liveSession: this.el.querySelector("#live-session"),
      livePid: this.el.querySelector("#live-pid"),
      liveSpan: this.el.querySelector("#live-span"),
      liveStep: this.el.querySelector("#live-step"),
      liveCommand: this.el.querySelector("#live-command"),
      liveError: this.el.querySelector("#live-error"),
    }

    liveDebug("hook mounted", {
      executionMode: this.executionMode,
      source: this.live.source,
      status: this.live.status,
    })

    this.handleEvent("trace_ready", payload => {
      this.executionMode = "replay"
      this.lessons = payload.lessons || this.lessons
      this.subjects = payload.subjects || this.subjects
      this.loadTrace(payload.trace)
    })

    this.handleEvent("execution_event", payload => {
      liveDebug("LiveView event received", payload)
      this.applyExecutionEvent(payload)
    })

    const initialTrace = parsePayload(this.el.dataset.trace, null)
    if (initialTrace && initialTrace.events && initialTrace.events.length > 0) {
      this.loadTrace(initialTrace)
    } else if (this.executionMode === "live") {
      this.renderLiveIdle()
    } else {
      this.renderLoading()
    }
  },

  destroyed() {
    this.stop()
    if (this.editor) this.editor.destroy()
    this.disposeGraph()
    window.removeEventListener("resize", this.resizeHandler)
    window.removeEventListener("keydown", this.keyHandler)
  },

  loadTrace(trace) {
    this.stop()
    this.disposeGraph()
    this.executionMode = "replay"
    if (this.editor) {
      this.editor.destroy()
      this.editor = null
    }

    this.trace = trace
    this.sourceById = new Map(this.trace.sources.map(source => [source.id, source]))
    this.eventBySourceLine = buildSourceLineIndex(this.trace.events)
    this.playback = initialPlayback(this.trace.events.length, readStepFromUrl())
    this.step = this.playback.step
    this.speed = this.playback.speed
    this.activeFile = null
    this.selected = null
    this.compareStep = null

    if (!this.controlsBound) {
      this.bindControls()
      this.controlsBound = true
    }

    this.initCode()
    this.initGraph()
    this.renderAll(false)
  },

  renderLoading() {
    this.refs.lessonTitle.textContent = "Loading trace"
    this.refs.codeTitle.textContent = "lesson.ex"
    this.refs.sourceTabs.innerHTML = ""
    this.refs.codeEditor.innerHTML = `<div class="loading-panel">Generating trace from the selected subject...</div>`
    this.refs.variableInspector.innerHTML = `<div class="inspector-title">Values</div>`
    this.refs.graphTitle.textContent = "Preparing graph"
    this.refs.phaseLabel.textContent = "Loading"
    this.refs.programSelector.innerHTML = this.renderProgramSelectorHtml(null)
    this.refs.progressStrip.innerHTML = ""
    this.refs.lessonCard.innerHTML = `<p class="lesson-copy">The backend is running the lesson and building an immutable timeline.</p>`
    this.refs.checkpointRow.innerHTML = ""
    this.refs.stepCounter.textContent = "Step - / -"
  },

  renderLiveIdle() {
    liveDebug("render live idle", this.live)
    this.stop()
    this.disposeGraph()
    if (!this.controlsBound) {
      this.bindControls()
      this.controlsBound = true
    }

    this.refs.lessonTitle.textContent = "Live AST execution"
    this.refs.graphTitle.textContent = "Live domain state"
    this.refs.phaseLabel.textContent = "Live"
    this.refs.programSelector.innerHTML = this.renderProgramSelectorHtml(this.currentLessonId)
    this.bindProgramSelector()
    this.refs.progressStrip.innerHTML = ""
    this.refs.lessonCard.innerHTML = `<p class="lesson-copy">Start live execution to pause the backend before the first expression.</p>`
    this.refs.checkpointRow.innerHTML = ""
    this.refs.stepCounter.textContent = "Live step 0"
    this.renderLiveStatus()
    this.renderLiveCode()
    this.renderLiveBindings()
    this.renderLiveGraph()
    this.renderLiveTeaching()
  },

  disposeGraph() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame)
      this.animationFrame = null
    }
    if (this.controls) {
      this.controls.dispose()
      this.controls = null
    }
    if (this.renderer) {
      this.renderer.dispose()
      this.renderer = null
    }
    this.scene = null
    this.camera = null
    this.nodeObjects = new Map()
    this.edgeObjects = new Map()
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

    this.el.querySelectorAll("[data-live-action]").forEach(button => {
      button.addEventListener("click", () => this.handleLiveAction(button.dataset.liveAction))
    })

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
      for (const object of this.edgeObjects.values()) object.label.visible = this.showLabels && this.showLocalDerivatives
    })

    this.el.querySelector("[data-toggle-derivatives]").addEventListener("click", event => {
      this.showLocalDerivatives = !this.showLocalDerivatives
      event.currentTarget.classList.toggle("active", this.showLocalDerivatives)
      this.renderGraph(false)
    })

    this.el.querySelector("[data-toggle-gradients]").addEventListener("click", event => {
      this.showGradients = !this.showGradients
      event.currentTarget.classList.toggle("active", this.showGradients)
      this.renderGraph(false)
    })

    this.el.querySelector("[data-reset-camera]").addEventListener("click", () => this.resetCamera())

    this.keyHandler = event => {
      if (event.target.matches("input, textarea, select")) return
      if (event.key === " ") {
        event.preventDefault()
        if (this.executionMode === "live") this.handleAction("toggle")
        else this.toggle()
      } else if (event.key === "ArrowRight") {
        event.preventDefault()
        if (this.executionMode === "live") {
          if (!event.shiftKey) this.handleLiveAction("step")
        } else {
          this.setStep(this.step + (event.shiftKey ? 10 : 1), true)
        }
      } else if (event.key === "ArrowLeft") {
        event.preventDefault()
        if (this.executionMode !== "live") this.setStep(this.step - (event.shiftKey ? 10 : 1), true)
      } else if (event.key === "Home") {
        event.preventDefault()
        if (this.executionMode === "live") this.handleLiveAction("start")
        else this.setStep(0, false)
      } else if (event.key === "End") {
        event.preventDefault()
        if (this.trace) this.setStep(this.trace.events.length - 1, false)
      } else if (/^[1-9]$/.test(event.key)) {
        if (!this.trace) return
        const checkpoint = this.trace.checkpoints[Number(event.key) - 1]
        if (checkpoint) this.setStep(checkpoint.step, false)
      }
    }
    window.addEventListener("keydown", this.keyHandler)
  },

  handleAction(action) {
    if (this.executionMode === "live") {
      switch (action) {
        case "start": return this.handleLiveAction("start")
        case "toggle": return this.handleLiveToggle()
        case "forward1": return this.handleLiveAction("step")
        case "end": return this.handleLiveAction("stop")
        default: return
      }
    }

    switch (action) {
      case "start": return this.setStep(0, false)
      case "back10": return this.setStep(this.step - 10, false)
      case "back1": return this.setStep(this.step - 1, true)
      case "toggle": return this.toggle()
      case "forward1": return this.setStep(this.step + 1, true)
      case "forward10": return this.setStep(this.step + 10, false)
      case "end": return this.setStep(this.trace.events.length - 1, false)
      case "loop_phase": return this.toggleLoopPhase()
      case "follow": return this.toggleFollow()
      case "compare": return this.toggleCompare()
    }
  },

  handleLiveAction(action) {
    if (!canSendLiveAction(this.live, action)) {
      liveDebug("blocked live action", {action, live: this.live})
      this.renderLiveStatus()
      return
    }

    const eventName = {
      start: "start_live",
      step: "step_live",
      continue: "continue_live",
      stop: "stop_live",
      reset: "reset_live",
    }[action]

    if (!eventName) return
    liveDebug("live action sent", {action, eventName, sessionId: this.live.sessionId})
    this.live = {...this.live, lastCommand: action, status: action === "start" ? "starting" : "command_sent"}
    this.renderLiveStatus()
    this.updateLiveControlState()
    this.pushEvent(eventName, {})
  },

  handleLiveToggle() {
    if (canSendLiveAction(this.live, "continue")) return this.handleLiveAction("continue")
    if (canSendLiveAction(this.live, "start")) return this.handleLiveAction("start")
    if (canSendLiveAction(this.live, "stop") && this.live.status === "running") return this.handleLiveAction("stop")
    this.renderLiveStatus()
  },

  applyExecutionEvent(payload) {
    this.executionMode = "live"
    this.live = reduceExecutionEvent(this.live, payload)
    if (this.live.source) this.code.lastKnownGoodSource = this.live.source
    this.currentSubjectId = payload.subject_id || this.currentSubjectId
    this.currentLessonId = payload.lesson_id || this.currentLessonId

    if (!this.controlsBound) {
      this.bindControls()
      this.controlsBound = true
    }

    this.renderLiveStatus()
    this.renderLiveCode()
    this.renderLiveBindings()
    this.renderLiveGraph()
    this.renderLiveTeaching(payload)
    this.updateLiveControlState()
  },

  toggleLoopPhase() {
    if (!this.trace) return
    this.playback = playbackReducer(this.playback, {type: "loop_phase"}, this.trace.events.length, this.trace.checkpoints)
    this.el.querySelectorAll("[data-action='loop_phase']").forEach(button => button.classList.toggle("active", this.playback.loopPhase))
  },

  toggleFollow() {
    if (!this.trace) return
    this.playback = playbackReducer(this.playback, {type: "follow"}, this.trace.events.length, this.trace.checkpoints)
    this.el.querySelectorAll("[data-action='follow']").forEach(button => button.classList.toggle("active", this.playback.follow))
    this.renderGraph(false)
  },

  toggleCompare() {
    if (!this.trace) return
    this.compareStep = this.compareStep == null ? this.step : null
    this.el.querySelectorAll("[data-action='compare']").forEach(button => button.classList.toggle("active", this.compareStep != null))
    this.renderSelection()
    this.renderTeaching()
  },

  toggle() {
    if (this.timer) this.stop()
    else this.play()
  },

  play() {
    if (!this.trace) return
    this.manualSourceFile = null
    this.el.querySelector("[data-action='toggle']").textContent = "Pause"
    const tick = () => {
      if (this.step >= this.trace.events.length - 1) {
        this.stop()
        return
      }
      this.setStep(nextLoopStep(this.playback, this.trace.events), true, false)
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
    if (!this.trace) return
    this.step = clamp(nextStep, 0, this.trace.events.length - 1)
    this.playback = playbackReducer(this.playback, {type: "jump", step: this.step}, this.trace.events.length, this.trace.checkpoints)
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
    this.refs.scrubber.max = Math.max(this.trace.events.length - 1, 0)
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
      updateTextSprite(object.label, labelText(object.node, grad, this.showGradients), 240, 82)
    }

    for (const [id, object] of this.edgeObjects) {
      const fromVisible = visible.has(String(object.edge.from))
      const toVisible = visible.has(String(object.edge.to))
      const isVisible = structural || (fromVisible && toVisible)
      const isActive = event.snapshot.active_edge === id
      object.line.visible = isVisible
      object.label.visible = isVisible && this.showLabels && this.showLocalDerivatives
      object.material.color.set(isActive ? "#16a34a" : "#44485f")
      object.material.opacity = isActive ? 1 : 0.55
    }

    this.renderMinimap(visible, structural)
    this.renderSelection()
    if (this.playback.follow) this.focusActiveNode(event.snapshot.active_node)
  },

  focusActiveNode(nodeId) {
    if (!this.controls || nodeId == null) return
    const pos = this.layout.get(String(nodeId))
    if (!pos) return
    this.controls.target.lerp(new THREE.Vector3(pos.x, pos.y, 0), 0.16)
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
      ${this.renderCompareFact()}
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

  renderCompareFact() {
    if (this.compareStep == null || !this.trace.events[this.compareStep]) return ""

    const from = this.trace.events[this.compareStep]
    const to = this.currentEvent()
    const fromNodes = new Set((from.snapshot.visible_nodes || []).map(String))
    const toNodes = new Set((to.snapshot.visible_nodes || []).map(String))
    const newNodes = [...toNodes].filter(id => !fromNodes.has(id)).length
    const gradientDelta = Object.keys(to.snapshot.gradients || {}).length - Object.keys(from.snapshot.gradients || {}).length

    return `
      <div class="compare-card">
        <strong>Compare step ${this.compareStep + 1} to ${this.step + 1}</strong>
        <span>${newNodes} new nodes, ${gradientDelta} gradient entries changed</span>
      </div>
    `
  },

  renderProgramSelector() {
    const current = this.trace.lesson_id
    this.refs.programSelector.innerHTML = this.renderProgramSelectorHtml(current)

    this.bindProgramSelector()
  },

  bindProgramSelector() {
    this.refs.programSelector.querySelectorAll("[data-subject-select]").forEach(select => {
      select.addEventListener("change", () => {
        const subject = this.subjects.find(item => item.id === select.value)
        const lesson = subject?.lessons?.[0]
        const params = new URLSearchParams(window.location.search)
        params.set("subject", select.value)
        if (lesson) params.set("lesson", lesson.id)
        if (this.executionMode === "live") params.set("mode", "live")
        params.delete("step")
        window.location.search = params.toString()
      })
    })

    this.refs.programSelector.querySelectorAll("[data-lesson]").forEach(button => {
      button.addEventListener("click", () => {
        const params = new URLSearchParams(window.location.search)
        if (this.executionMode === "live") {
          params.set("subject", this.currentSubjectId)
          params.set("mode", "live")
        } else {
          params.set("subject", this.trace.subject_id)
        }
        params.set("lesson", button.dataset.lesson)
        params.delete("step")
        window.location.search = params.toString()
      })
    })
  },

  renderProgramSelectorHtml(current) {
    const groups = new Map()
    for (const lesson of this.lessons) {
      if (!groups.has(lesson.level)) groups.set(lesson.level, [])
      groups.get(lesson.level).push(lesson)
    }

    const selectedSubject = this.trace?.subject_id || this.live?.subject_id || this.currentSubjectId
    const subjectOptions = (this.subjects || []).map(subject => `
      <option value="${escapeHtml(subject.id)}" ${subject.id === selectedSubject ? "selected" : ""}>
        ${escapeHtml(subject.title)}
      </option>
    `).join("")

    const subjectSelector = subjectOptions
      ? `<label class="subject-select"><span>Subject</span><select data-subject-select>${subjectOptions}</select></label>`
      : ""

    return subjectSelector + Array.from(groups.entries()).map(([level, lessons]) => `
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
          ${this.renderCompareFact()}
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
          ${this.renderCompareFact()}
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
    writeStepToUrl(this.trace.lesson_id, this.step, this.trace.subject_id)
  },

  currentEvent() {
    return this.trace.events[this.step]
  },

  renderLiveStatus() {
    if (!this.refs.executionStatusCard) return
    const span = this.live.span
    this.refs.executionStatusCard.querySelector(".status-pill").textContent = "LIVE"
    this.refs.executionStatusCard.querySelector("strong").textContent = "Live AST execution"
    this.refs.liveStatus.textContent = this.live.status || "idle"
    this.refs.liveSession.textContent = this.live.sessionId || "-"
    this.refs.livePid.textContent = this.live.runtimePid || "-"
    this.refs.liveSpan.textContent = span ? `${span.file || "lesson.ex"}:${span.line_start || span.line || 1}` : "-"
    this.refs.liveStep.textContent = String(this.live.currentStep || 0)
    this.refs.liveCommand.textContent = this.live.lastCommand || "-"
    this.refs.stepCounter.textContent = `Live step ${this.live.currentStep || 0}`
    if (this.refs.liveError) {
      const message = this.live.error?.message || ""
      this.refs.liveError.hidden = !message
      this.refs.liveError.textContent = message
    }
    this.updateLiveControlState()
  },

  renderLiveCode() {
    const source = normalizeSource(this.live.source) ||
      normalizeSource(parsePayload(this.el.dataset.liveSource, null)) ||
      this.preserveLastKnownSource()

    if (!source) {
      this.renderCodeError({message: "Live source unavailable for this lesson."})
      return
    }

    this.setCodeSource(source)
    this.setActiveSpan(this.live.span)
  },

  ensureCodeEditor(source = this.code.lastKnownGoodSource) {
    const normalized = normalizeSource(source)
    if (!normalized) return null

    const sourceId = `live:${normalized.id || normalized.file || "lesson.ex"}`
    const editorDetached = this.editor && this.editor.dom.parentNode !== this.refs.codeEditor
    const sourceChanged = this.activeFile !== sourceId ||
      this.editor?.state?.doc?.toString() !== normalized.source

    if (!this.editor || editorDetached || sourceChanged) {
      if (this.editor) this.editor.destroy()
      this.refs.codeEditor.innerHTML = ""
      this.activeFile = sourceId
      this.editor = new EditorView({
        state: EditorState.create({
          doc: normalized.source,
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
      liveDebug("CodeMirror source set", {sourceId, length: normalized.source.length})
    }

    return this.editor
  },

  setCodeSource(sourcePayload) {
    const source = normalizeSource(sourcePayload)
    if (!source) {
      this.renderCodeError({message: "Malformed live source payload."})
      return
    }

    this.code.lastKnownGoodSource = source
    this.code.error = null
    this.refs.codeTitle.textContent = source.title || "lesson.ex"
    this.refs.sourceTabs.innerHTML = `<button type="button" class="source-tab active">${escapeHtml(source.title || "lesson.ex")}</button>`
    this.ensureCodeEditor(source)
  },

  setActiveSpan(span) {
    const editor = this.ensureCodeEditor()
    if (!editor) return

    const line = span?.line_start || span?.line || 1
    this.code.activeLineStart = line
    this.code.activeLineEnd = span?.line_end || line
    editor.dispatch({effects: activeLineEffect.of(line)})
    const codeLine = editor.state.doc.line(Math.min(line, editor.state.doc.lines))
    editor.dispatch({selection: {anchor: codeLine.from}, scrollIntoView: true})
  },

  renderCodeError(error) {
    this.code.error = error
    const source = this.preserveLastKnownSource()
    if (source) {
      this.setCodeSource(source)
      this.setActiveSpan(this.live.span)
      liveDebug("source preservation on error", error)
      return
    }

    this.refs.codeTitle.textContent = "lesson.ex"
    this.refs.sourceTabs.innerHTML = ""
    this.refs.codeEditor.innerHTML = `<div class="loading-panel">${escapeHtml(error?.message || "Live source unavailable.")}</div>`
  },

  preserveLastKnownSource() {
    return normalizeSource(this.code.lastKnownGoodSource)
  },

  renderLiveBindings() {
    const rows = bindingRows(this.live.bindings).map(row => `
      <div class="var-row live-binding" data-binding="${escapeHtml(row.name)}">
        <span>${escapeHtml(row.name)}</span>
        <b>${escapeHtml(row.kind || "")}</b>
        <em>${escapeHtml(row.summary || "")}</em>
      </div>
    `).join("")

    this.refs.variableInspector.innerHTML = `<div class="inspector-title">Live bindings</div>${rows || `<p class="empty-copy">No bindings yet</p>`}`
  },

  renderLiveTeaching(event = {}) {
    const span = this.live.span
    const phase = this.live.domainSnapshot?.phase || livePhase(this.live)
    const focused = this.live.domainSnapshot?.active_value_name || this.live.domainSnapshot?.activeNodeId
    this.refs.lessonTitle.textContent = `${this.currentLessonId} live`
    this.refs.lessonCard.innerHTML = `
      <div class="lesson-card-header">
        <span>${escapeHtml(this.live.status || "idle")}</span>
        <strong>${escapeHtml(event.type || "Live AST execution")}</strong>
      </div>
      <dl class="fact-grid">
        <div><dt>mode</dt><dd>live</dd></div>
        <div><dt>step</dt><dd>${escapeHtml(String(this.live.currentStep || 0))}</dd></div>
        <div><dt>phase</dt><dd>${escapeHtml(phase)}</dd></div>
        <div><dt>focus</dt><dd>${escapeHtml(focused || "-")}</dd></div>
      </dl>
      <p class="lesson-copy">${escapeHtml(liveExplanation(this.live, event, this.currentLessonId))}</p>
      ${span ? `<p class="lesson-copy">Source span: ${escapeHtml(span.file || "lesson.ex")}:${span.line_start || span.line || 1}</p>` : ""}
      ${event.error ? `<p class="lesson-copy">${escapeHtml(event.error.message || String(event.error))}</p>` : ""}
    `
  },

  renderLiveGraph() {
    const canvas = this.refs.canvas
    const wrap = canvas.parentElement
    const width = Math.max(wrap.clientWidth, 300)
    const height = Math.max(wrap.clientHeight, 260)
    canvas.width = Math.floor(width * Math.min(window.devicePixelRatio || 1, 2))
    canvas.height = Math.floor(height * Math.min(window.devicePixelRatio || 1, 2))
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
    const ctx = canvas.getContext("2d")
    const scale = canvas.width / width
    ctx.setTransform(scale, 0, 0, scale, 0, 0)
    ctx.clearRect(0, 0, width, height)
    ctx.fillStyle = "#0e0e12"
    ctx.fillRect(0, 0, width, height)

    const domain = this.live.domainSnapshot
    if (!domain?.graph?.nodes?.length) {
      ctx.fillStyle = "#8888aa"
      ctx.font = "14px ui-sans-serif, system-ui"
      ctx.fillText("Live graph appears as execution reaches Micrograd values.", 24, 36)
      this.refs.minimap.innerHTML = `<span class="live-mini-row">No live nodes yet</span>`
      return
    }

    const nodes = domain.graph.nodes
    const edges = domain.graph.edges || []
    this.refs.phaseLabel.textContent = titleCase(domain.phase || livePhase(this.live))
    this.refs.graphTitle.textContent = "Live Micrograd graph"
    this.refs.minimap.innerHTML = nodes.map(node => {
      const grad = domain.gradients?.[String(node.id)] ?? domain.gradients?.[node.id]
      return `<span class="live-mini-row">${escapeHtml(node.title || node.display_id)} ${formatNumber(node.data)} grad ${grad == null ? "-" : formatNumber(grad)}</span>`
    }).join("")
    const positions = new Map()
    const spacing = Math.max(120, Math.min(180, (width - 80) / Math.max(nodes.length, 1)))
    nodes.forEach((node, index) => positions.set(String(node.id), {x: 50 + index * spacing, y: height / 2}))

    for (const edge of edges) {
      const from = positions.get(String(edge.from))
      const to = positions.get(String(edge.to))
      if (!from || !to) continue
      ctx.strokeStyle = "#16a34a"
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(from.x + 36, from.y)
      ctx.lineTo(to.x - 36, to.y)
      ctx.stroke()
      ctx.fillStyle = "#9ca3af"
      ctx.font = "12px ui-monospace, monospace"
      ctx.fillText(edge.label || "", (from.x + to.x) / 2 - 12, from.y - 16)
    }

    for (const node of nodes) {
      const pos = positions.get(String(node.id))
      const grad = domain.gradients?.[String(node.id)] ?? domain.gradients?.[node.id]
      ctx.fillStyle = node.kind === "leaf" ? "#4b50c8" : "#c8930a"
      ctx.strokeStyle = node.is_output ? "#ffffff" : "#252638"
      ctx.lineWidth = node.is_output ? 3 : 1
      roundRect(ctx, pos.x - 44, pos.y - 28, 88, 56, 8)
      ctx.fill()
      ctx.stroke()
      ctx.fillStyle = "#f5f5fb"
      ctx.font = "12px ui-monospace, monospace"
      ctx.textAlign = "center"
      ctx.fillText(String(node.title || node.display_id).slice(0, 14), pos.x, pos.y - 8)
      ctx.fillText(`data ${formatNumber(node.data)}`, pos.x, pos.y + 8)
      ctx.fillText(`grad ${grad == null ? "-" : formatNumber(grad)}`, pos.x, pos.y + 23)
    }
    ctx.textAlign = "left"
  },

  updateLiveControlState() {
    if (this.executionMode !== "live") return

    for (const button of this.el.querySelectorAll("[data-live-action]")) {
      const action = button.dataset.liveAction
      button.disabled = !canSendLiveAction(this.live, action)
      button.setAttribute("aria-disabled", String(button.disabled))
    }

    const cinemaState = {
      start: canSendLiveAction(this.live, "start"),
      back10: false,
      back1: false,
      toggle: canSendLiveAction(this.live, "start") ||
        canSendLiveAction(this.live, "continue") ||
        (this.live.status === "running" && canSendLiveAction(this.live, "stop")),
      forward1: canSendLiveAction(this.live, "step"),
      forward10: false,
      end: canSendLiveAction(this.live, "stop"),
      loop_phase: false,
      follow: true,
      compare: false,
    }

    for (const button of this.el.querySelectorAll("[data-action]")) {
      const action = button.dataset.action
      if (!(action in cinemaState)) continue
      button.disabled = !cinemaState[action]
      button.setAttribute("aria-disabled", String(button.disabled))
      if (action === "toggle") {
        button.textContent = canSendLiveAction(this.live, "start")
          ? "Start"
          : this.live.status === "running"
            ? "Stop"
            : "Play"
      }
    }

    if (this.refs.scrubber) this.refs.scrubber.disabled = true
    if (this.refs.speedSelect) this.refs.speedSelect.disabled = true
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

function labelText(node, grad, showGradients = true) {
  const title = node.title || node.display_id
  const gradText = showGradients ? `\ngrad ${grad == null ? "-" : formatNumber(grad)}` : ""
  return `${title}\n${node.op}  data ${formatNumber(node.data)}${gradText}`
}

function livePhase(live) {
  if (!live.sessionId) return "pre_run"
  if (live.status === "completed") return "completed"
  if (live.bindings?.gradients) return "backward"
  if (live.bindings?.y) return "forward"
  if (live.bindings?.x) return "forward"
  return "pre_run"
}

function liveExplanation(live, event = {}, lessonId = "x_squared") {
  if (event?.type === "command_error") {
    return event.error?.message || event.message || "That live command is not valid for the current backend state."
  }

  if (lessonId !== "x_squared") {
    if (!live.sessionId) return "Start live execution to create a backend session for this lesson."
    if (live.status === "paused") return "The backend is paused at the current source span and waiting for the next command."
    if (live.status === "completed") return "The backend runtime completed the configured lesson source."
    return "Live execution is synchronized from backend runtime events."
  }

  if (!live.sessionId) {
    return "This lesson will create x, compute y = x^2, then run backward."
  }

  const line = live.span?.line_start || live.span?.line
  const hasX = Boolean(live.bindings?.x)
  const hasY = Boolean(live.bindings?.y)
  const hasGradients = Boolean(live.bindings?.gradients)

  if (live.status === "completed" || hasGradients) {
    return "The gradient table now says dy/dx = 6."
  }
  if (line === 3 && hasY) return "The next expression starts reverse-mode autodiff."
  if (hasY) return "y is an operation node with data 9 and a dependency on x."
  if (line === 2 && hasX) return "The next expression computes y = x^2."
  if (hasX) return "x now exists as a scalar node with data 3."
  if (line === 1) return "The next expression creates a leaf scalar Value."
  return "The backend is paused at the current source span and waiting for the next command."
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

export {VizLabHook}
