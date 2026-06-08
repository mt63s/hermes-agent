/**
 * Phase 1 render test (spec v4 §5 Layer 2). Mounts the App headlessly with a
 * store seeded by the scripted hello stream, asserts the captured frame is
 * THEMED (brand name/icon from the theme, not hardcoded), and that applying a
 * custom skin re-themes the brand name reactively.
 */
import { describe, expect, test } from 'bun:test'

import { createSessionStore } from '../logic/store.ts'
import { App } from '../view/App.tsx'
import { ThemeProvider } from '../view/theme.tsx'
import { captureFrame } from './lib/render.ts'

function seedHello(store: ReturnType<typeof createSessionStore>) {
  store.apply({ type: 'gateway.ready' })
  store.apply({ type: 'message.start' })
  store.apply({ type: 'message.delta', payload: { text: 'Hi there, glitch!' } })
  store.apply({ type: 'message.complete' })
}

describe('App render (Phase 1, themed)', () => {
  test('renders the streamed hello + default brand into the frame', async () => {
    const store = createSessionStore()
    seedHello(store)

    const frame = await captureFrame(
      () => (
        <ThemeProvider theme={() => store.state.theme}>
          <App store={store} />
        </ThemeProvider>
      ),
      { width: 60, height: 16 }
    )

    expect(frame).toContain('Hermes Agent') // default brand.name
    expect(frame).toContain('ready')
    expect(frame).toContain('Hi there, glitch!')
    expect(frame).toContain('Type your message') // composer placeholder (brand.welcome)
  })

  test('applying a skin re-themes the brand name (skinnable, no hardcoding)', async () => {
    const store = createSessionStore()
    store.apply({ type: 'gateway.ready', payload: { skin: { branding: { agent_name: 'Zephyr' } } } })
    seedHello(store)

    const frame = await captureFrame(
      () => (
        <ThemeProvider theme={() => store.state.theme}>
          <App store={store} />
        </ThemeProvider>
      ),
      { width: 60, height: 16 }
    )

    expect(frame).toContain('Zephyr')
    expect(frame).not.toContain('Hermes Agent')
  })

  test('renders an inline tool part between text (ordered parts §7)', async () => {
    const store = createSessionStore()
    store.apply({ type: 'gateway.ready' })
    store.apply({ type: 'message.start' })
    store.apply({ type: 'message.delta', payload: { text: 'Listing files:' } })
    store.apply({ type: 'tool.start', payload: { tool_id: 't1', name: 'terminal' } })
    store.apply({
      type: 'tool.complete',
      payload: { tool_id: 't1', result_text: '{"output":"alpha.txt\\nbeta.txt","exit_code":0}' }
    })
    store.apply({ type: 'message.complete' })

    const frame = await captureFrame(
      () => (
        <ThemeProvider theme={() => store.state.theme}>
          <App store={store} />
        </ThemeProvider>
      ),
      { width: 60, height: 16 }
    )

    expect(frame).toContain('Listing files:') // text part
    expect(frame).toContain('terminal') // tool name (inline, between text blocks)
    expect(frame).toContain('alpha.txt') // envelope-stripped output, block-rendered
    expect(frame).not.toContain('exit_code') // the {output,exit_code} envelope is stripped
  })
})
