/**
 * ToolPart — compact inline/block render of one tool call (spec v4 §7 / §8;
 * ported from the React build's `ToolRow`). Two tiers:
 *   - running, or complete with ≤1 output line → a one-line row `⚡ name  status`
 *   - complete with multi-line output → a left-bar block (NOT a full box), capped
 *     to TOOL_MAX_LINES with a "… +N more" marker (click-to-expand when mouse is on)
 * `resultText` is already `{output,exit_code}`-envelope-stripped by the store.
 * Fully themed (no hardcoded styles).
 */
import { type ToolPartState } from '../logic/store.ts'
import { useTerminalDimensions } from '@opentui/solid'
import { createMemo, createSignal, For, Match, Show, Switch } from 'solid-js'

import { collapseToolOutput, truncate } from '../logic/toolOutput.ts'
import { useTheme } from './theme.tsx'

const GUTTER = 2
const TOOL_MAX_LINES = 10

export function ToolPart(props: { part: ToolPartState }) {
  const theme = useTheme()
  const dims = useTerminalDimensions()
  const [expanded, setExpanded] = createSignal(false)

  const bodyWidth = () => Math.max(20, dims().width - GUTTER - 4)
  const result = () => (props.part.resultText ?? '').replace(/\s+$/, '')
  const lines = () => (result() ? result().split('\n') : [])
  const multiline = () => lines().length > 1
  const inlineStatus = () => (props.part.error ? `✗ ${props.part.error}` : (lines()[0] ?? props.part.summary ?? ''))
  const collapsed = createMemo(() =>
    collapseToolOutput(result(), expanded() ? lines().length : TOOL_MAX_LINES, bodyWidth() - 2)
  )

  return (
    <box style={{ flexDirection: 'row', flexShrink: 0, marginTop: 1 }}>
      <box style={{ flexShrink: 0, width: GUTTER }}>
        <text>
          <span style={{ fg: theme().color.muted }}>⚡</span>
        </text>
      </box>
      <Switch>
        <Match when={props.part.state === 'running'}>
          <text>
            <span style={{ fg: theme().color.label }}>{props.part.name}</span>
            <span style={{ fg: theme().color.muted }}> …</span>
          </text>
        </Match>
        <Match when={!multiline()}>
          <text>
            <span style={{ fg: theme().color.label }}>{props.part.name}</span>
            <Show when={inlineStatus()}>
              <span style={{ fg: props.part.error ? theme().color.error : theme().color.muted }}>
                {`  ${truncate(inlineStatus(), Math.max(1, bodyWidth() - props.part.name.length - 2))}`}
              </span>
            </Show>
          </text>
        </Match>
        <Match when={multiline()}>
          <box style={{ flexDirection: 'row', flexGrow: 1, minWidth: 0 }} onMouseDown={() => setExpanded(e => !e)}>
            <box
              style={{
                backgroundColor: props.part.error ? theme().color.error : theme().color.border,
                flexShrink: 0,
                width: 1
              }}
            />
            <box style={{ flexDirection: 'column', flexGrow: 1, minWidth: 0, paddingLeft: 1 }}>
              <text>
                <span style={{ fg: theme().color.label }}>{props.part.name}</span>
              </text>
              <For each={collapsed().lines}>
                {line => (
                  <text>
                    <span style={{ fg: theme().color.muted }}>{line}</span>
                  </text>
                )}
              </For>
              <Show when={collapsed().hiddenLines > 0}>
                <text>
                  <span style={{ fg: theme().color.accent }}>
                    {`… +${collapsed().hiddenLines} more line${collapsed().hiddenLines === 1 ? '' : 's'}`}
                  </span>
                </text>
              </Show>
              <Show when={props.part.error}>
                <text>
                  <span style={{ fg: theme().color.error }}>{truncate(props.part.error ?? '', bodyWidth() - 2)}</span>
                </text>
              </Show>
            </box>
          </box>
        </Match>
      </Switch>
    </box>
  )
}
