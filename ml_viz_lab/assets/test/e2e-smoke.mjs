import {chromium} from "playwright"
import {PNG} from "pngjs"

const baseUrl = process.env.BASE_URL || "http://localhost:4000"
const lessons = [
  "x_squared",
  "sum_rule",
  "chain_rule_composed",
  "karpathy_sanity",
  "repeated_parent",
  "relu_gate",
  "single_neuron",
  "one_layer_forward",
  "tiny_mlp",
  "mlp_loss_backward",
  "one_training_step",
  "linear_training",
]

const browser = await chromium.launch({headless: true})

try {
  for (const lesson of lessons) {
    const page = await browser.newPage({viewport: {width: 1360, height: 860}})
    const errors = []
    page.on("pageerror", error => errors.push(error.message))
    page.on("console", message => {
      if (message.type() === "error") errors.push(message.text())
    })

    await page.goto(`${baseUrl}/?lesson=${lesson}`, {waitUntil: "networkidle"})
    await page.waitForSelector("#viz-lab")
    await page.waitForFunction(() => document.querySelector("#step-counter")?.textContent?.includes(" / "))
    await page.waitForTimeout(500)

    const title = await page.locator("#lesson-title").innerText()
    const before = await page.locator("#step-counter").innerText()
    await page.locator("[data-action='forward1']").click()
    await page.waitForTimeout(150)
    const after = await page.locator("#step-counter").innerText()
    await page.locator("[data-toggle-gradients]").click()
    await page.locator("[data-toggle-derivatives]").click()
    await page.locator("[data-action='compare']").click()
    await page.locator("[data-action='end']").click()
    await page.waitForTimeout(150)
    const end = await page.locator("#step-counter").innerText()
    const canvas = await page.locator("#graph-canvas").boundingBox()

    if (errors.length) throw new Error(`${lesson}: ${errors.join("\n")}`)
    if (before === after) throw new Error(`${lesson}: step did not advance from ${before}`)
    if (!canvas || canvas.width < 300 || canvas.height < 250) {
      throw new Error(`${lesson}: bad canvas ${JSON.stringify(canvas)}`)
    }
    const canvasPng = PNG.sync.read(await page.locator("#graph-canvas").screenshot())
    const colors = sampledColorCount(canvasPng)
    if (colors < 3) throw new Error(`${lesson}: canvas appears blank or too flat (${colors} colors)`)

    console.log(`${lesson}: ${title} ${before} -> ${end}`)
    await page.close()
  }
} finally {
  await browser.close()
}

function sampledColorCount(png) {
  const colors = new Set()
  const stride = Math.max(4, Math.floor((png.width * png.height) / 900) * 4)

  for (let index = 0; index < png.data.length; index += stride) {
    const red = png.data[index]
    const green = png.data[index + 1]
    const blue = png.data[index + 2]
    const alpha = png.data[index + 3]
    if (alpha === 0) continue
    colors.add(`${red >> 4}:${green >> 4}:${blue >> 4}`)
  }

  return colors.size
}
