import { memo, useEffect, useMemo, useRef } from 'react'

import { $petState, type PetInfo, type PetState } from '@/store/pet'

const DEFAULT_FRAME_W = 192
const DEFAULT_FRAME_H = 208
const DEFAULT_FRAMES = 6
const DEFAULT_LOOP_MS = 1100
// Mirrors agent.pet.constants.DEFAULT_SCALE — fallback only; the gateway sends
// the configured scale.
const DEFAULT_SCALE = 0.33
const DEFAULT_STATE_ROWS = ['idle', 'wave', 'run', 'failed', 'review', 'jump', 'extra1', 'extra2']

interface PetSpriteProps {
  info: PetInfo
  /** On-screen scale multiplier applied on top of the pet's native scale. */
  zoom?: number
}

/**
 * Canvas renderer for a petdex spritesheet — the one piece that must be
 * TypeScript (the engine's decode/encode is Python). Draws the row matching the
 * live `$petState`, stepping `framesPerState` frames across a `loopMs` loop.
 *
 * State is read from `$petState` via a ref + subscription rather than a prop,
 * so the frequent activity-driven state changes during an agent turn update the
 * canvas (inside its RAF loop) WITHOUT triggering a React re-render. Combined
 * with `memo`, this component effectively never re-renders after mount until
 * the pet itself changes.
 */
function PetSpriteImpl({ info, zoom = 1 }: PetSpriteProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const stateRef = useRef<PetState>($petState.get())

  const frameW = info.frameW ?? DEFAULT_FRAME_W
  const frameH = info.frameH ?? DEFAULT_FRAME_H
  const frames = info.framesPerState ?? DEFAULT_FRAMES
  const loopMs = info.loopMs ?? DEFAULT_LOOP_MS
  const scale = (info.scale ?? DEFAULT_SCALE) * zoom
  const rows = info.stateRows ?? DEFAULT_STATE_ROWS

  const drawW = Math.round(frameW * scale)
  const drawH = Math.round(frameH * scale)

  const image = useMemo(() => {
    if (!info.spritesheetBase64) {
      return null
    }

    const img = new Image()
    img.src = `data:${info.mime ?? 'image/webp'};base64,${info.spritesheetBase64}`

    return img
  }, [info.spritesheetBase64, info.mime])

  useEffect(() => {
    const canvas = canvasRef.current

    if (!canvas || !image) {
      return
    }

    const ctx = canvas.getContext('2d')

    if (!ctx) {
      return
    }

    // Track state via subscription, not a prop — no re-render on activity ticks.
    stateRef.current = $petState.get()

    const unsubState = $petState.listen(next => {
      stateRef.current = next
    })

    let raf = 0
    let frame = 0
    let lastStep = performance.now()
    let drawnFrame = -1
    let drawnRow = -1
    const stepMs = loopMs / Math.max(1, frames)

    const rowIndex = (s: PetState) => {
      const idx = rows.indexOf(s)

      return idx >= 0 ? idx : 0
    }

    const render = (now: number) => {
      if (now - lastStep >= stepMs) {
        frame = (frame + 1) % Math.max(1, frames)
        lastStep = now
      }

      const row = rowIndex(stateRef.current)

      // Only touch the canvas when the visible cell actually changes. The RAF
      // ticks at ~60Hz but the sprite only steps ~5Hz, so this skips ~90% of
      // the clear+draw work and keeps the main thread free.
      if ((frame !== drawnFrame || row !== drawnRow) && image.complete && image.naturalWidth > 0) {
        const sheetCols = Math.max(1, Math.floor(image.width / frameW))
        const sx = (frame % sheetCols) * frameW
        const sy = row * frameH
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.imageSmoothingEnabled = false
        ctx.drawImage(image, sx, sy, frameW, frameH, 0, 0, drawW, drawH)
        drawnFrame = frame
        drawnRow = row
      }

      raf = requestAnimationFrame(render)
    }

    raf = requestAnimationFrame(render)

    return () => {
      cancelAnimationFrame(raf)
      unsubState()
    }
  }, [image, frameW, frameH, frames, loopMs, drawW, drawH, rows])

  return (
    <canvas
      aria-label={info.displayName ? `${info.displayName} pet` : 'pet'}
      height={drawH}
      ref={canvasRef}
      style={{ height: drawH, width: drawW }}
      width={drawW}
    />
  )
}

/**
 * Memoized so a parent re-render (e.g. a position commit on drag-end) doesn't
 * re-run the canvas setup. Props change only when the pet itself changes.
 */
export const PetSprite = memo(PetSpriteImpl)
