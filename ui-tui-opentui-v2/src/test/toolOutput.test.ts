/**
 * toolOutput unit test (spec v4 §5 Layer 4 — Hermes-specific contract). The
 * `{output,exit_code}` envelope unwrap + the line/char collapse, as pure data.
 */
import { describe, expect, test } from 'bun:test'

import { collapseToolOutput, stripToolEnvelope, truncate } from '../logic/toolOutput.ts'

describe('stripToolEnvelope', () => {
  test('unwraps {output,exit_code} → output', () => {
    expect(stripToolEnvelope('{"output":"hi","exit_code":0}')).toBe('hi')
  })
  test('appends an [exit N] suffix on non-zero exit', () => {
    expect(stripToolEnvelope('{"output":"oops","exit_code":2}')).toBe('oops\n[exit 2]')
  })
  test('appends an [error] suffix when error is set', () => {
    expect(stripToolEnvelope('{"output":"x","error":"boom"}')).toBe('x\n[error] boom')
  })
  test('passes through non-JSON / non-envelope unchanged', () => {
    expect(stripToolEnvelope('just text')).toBe('just text')
    expect(stripToolEnvelope('{not json')).toBe('{not json')
    expect(stripToolEnvelope('{"result":"no output key"}')).toBe('{"result":"no output key"}')
  })
})

describe('collapseToolOutput / truncate', () => {
  test('caps to maxLines and reports the hidden count', () => {
    const c = collapseToolOutput('a\nb\nc\nd', 2, 10)
    expect(c.lines).toEqual(['a', 'b'])
    expect(c.hiddenLines).toBe(2)
    expect(c.truncated).toBe(true)
  })
  test('no truncation when within the cap', () => {
    const c = collapseToolOutput('a\nb', 5, 10)
    expect(c.lines).toEqual(['a', 'b'])
    expect(c.hiddenLines).toBe(0)
    expect(c.truncated).toBe(false)
  })
  test('truncate adds an ellipsis only when cut', () => {
    expect(truncate('abcdef', 4)).toBe('abc…')
    expect(truncate('ab', 4)).toBe('ab')
  })
})
