// WCAG AA contrast audit. Visits the dashboard's main surfaces in both themes
// and reports color-contrast violations from axe-core.
//
// Usage:
//   node scripts/contrast-audit.mjs                    # default: http://localhost:5173
//   BASE_URL=http://localhost:22089 node scripts/contrast-audit.mjs
//   TABLE_PATH=foo.bar node scripts/contrast-audit.mjs
//
// Requires: dashboard dev server (or built bundle) running at BASE_URL,
// and the Pixeltable API server reachable so /api/* responds.

import { chromium } from '@playwright/test'
import { AxeBuilder } from '@axe-core/playwright'
import { writeFileSync } from 'node:fs'

const BASE_URL = process.env.BASE_URL ?? 'http://localhost:5173'
const TABLE_PATH = process.env.TABLE_PATH ?? 'demo/frames'

const ROUTES = [
  { path: '/', name: 'home' },
  { path: `/table/${TABLE_PATH}`, name: `table-${TABLE_PATH.replace(/\//g, '-')}` },
  { path: '/lineage', name: 'lineage' },
]

const THEMES = ['light', 'dark']

async function setTheme(page, theme) {
  // Theme is persisted in localStorage('pxt-theme') and applied as <html class="dark">.
  await page.addInitScript((t) => {
    try { localStorage.setItem('pxt-theme', t) } catch {}
  }, theme)
}

async function runOne(browser, route, theme) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
  const page = await ctx.newPage()
  await setTheme(page, theme)
  await page.goto(`${BASE_URL}${route.path}`, { waitUntil: 'networkidle' })
  // Give late content (data fetches, lazy panels) a beat to render.
  await page.waitForTimeout(800)

  const results = await new AxeBuilder({ page })
    .options({ runOnly: { type: 'rule', values: ['color-contrast'] } })
    .analyze()

  await ctx.close()
  return results.violations
}

function summarize(violations) {
  // Each violation has a .nodes[] with target selectors and a failureSummary.
  // We extract the per-node measurement (fg, bg, ratio, expected) when present.
  const out = []
  for (const v of violations) {
    for (const n of v.nodes) {
      const data = n.any?.[0]?.data ?? {}
      out.push({
        impact: n.impact,
        selector: n.target?.join(' >> ') ?? '',
        snippet: (n.html ?? '').slice(0, 200),
        fg: data.fgColor,
        bg: data.bgColor,
        ratio: data.contrastRatio,
        expected: data.expectedContrastRatio,
        fontSize: data.fontSize,
        fontWeight: data.fontWeight,
      })
    }
  }
  return out
}

async function main() {
  const browser = await chromium.launch()
  const report = {}

  for (const theme of THEMES) {
    for (const route of ROUTES) {
      const key = `${theme}::${route.name}`
      process.stdout.write(`auditing ${key} ... `)
      try {
        const violations = await runOne(browser, route, theme)
        const flat = summarize(violations)
        report[key] = flat
        process.stdout.write(`${flat.length} violations\n`)
      } catch (e) {
        report[key] = { error: String(e) }
        process.stdout.write(`ERROR: ${e}\n`)
      }
    }
  }

  await browser.close()

  const outPath = 'contrast-audit.json'
  writeFileSync(outPath, JSON.stringify(report, null, 2))
  console.log(`\nWrote ${outPath}`)

  // Console summary table.
  console.log('\nSummary:')
  for (const [key, val] of Object.entries(report)) {
    const n = Array.isArray(val) ? val.length : 'ERR'
    console.log(`  ${key.padEnd(40)} ${n}`)
  }
}

main()
