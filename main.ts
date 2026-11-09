// Roblox Top Trending forecast (make.fun roblox-trending-forecast-01).
//
// Part 1, matrix model: a Markov chain over chart ranks. Each state is a rank
// 1..20 in an earlier read; the next state is that game's rank in a later read
// (1..20) or OUT (index 20, outside the top 20). Transitions come from every
// read pair whose gap is 0.4..16 hours, smoothed and row-normalised. The matrix
// for a horizon of h hours is P^x with x = min((h / medianGap)^0.3, 2.0),
// blended linearly between integer powers. The forecast averages those matrices
// over the 24 hourly horizons. Candidates are the games ranked 1..20 in the last
// read, scored for places 1..10; each place's probabilities are scaled so they
// sum to at most 0.85. This gives mapMatrix.
//
// Part 2, shift model: for each place 1..10, the games at ranks place-1..place+3
// of the last read get weights 0.10, 0.30, 0.30, 0.135, 0.06 (ranks the chart
// does not have are skipped). Those are normalised to 0.75 per place. The other
// 0.25 is hedge, spread over ranks 11..60 (or the chart's length) with weights
// exp(-(rank - 11) / 16). This gives mapShift.
//
// Blend: for every (game, place) key in either map, p = 0.4 * mapMatrix +
// 0.6 * mapShift. Values below 0.002 are dropped, rows are sorted by place then
// universe_id, and any place whose rows still sum above 1 is scaled down to 1.
// top10 is the last read's top 10.

import { mkdirSync, readFileSync, existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const PLACES = 10;
const RANKS = 20;
const OUT = RANKS;
const STATES = RANKS + 1;
const KEEP_READS = 240;
const GAP_MIN_H = 0.4;
const GAP_MAX_H = 16;
const ALPHA = 0.2;
const DEFAULT_MEDIAN_GAP = 6.4;
const MAX_MULTIPLICATIONS = 40;
const MIN_P = 0.002;
const PLACE_CAP = 0.85;
const SHIFTS: [number, number][] = [
  [-1, 0.1],
  [0, 0.3],
  [1, 0.3],
  [2, 0.135],
  [3, 0.06],
];
const HEDGE = 0.25;
const HEDGE_FROM = 11;
const HEDGE_TO = 60;
const HEDGE_DECAY = 16;
const MATRIX_BLEND = 0.4;
const SHIFT_BLEND = 0.6;

type Read = { at: number; top: number[]; rankOf: Map<number, number> };
type Matrix = number[][];
type Row = { universe_id: number; place: number; p: number };

const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

function loadReads(file: string): Read[] {
  if (!existsSync(file)) return [];
  const reads: Read[] = [];
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const obj = JSON.parse(line);
    if (!Array.isArray(obj.top_trending)) continue;
    const entries = [...obj.top_trending].sort((a: any, b: any) => a.rank - b.rank);
    reads.push({
      at: Date.parse(obj.at),
      top: entries.map((e: any) => e.universe_id),
      rankOf: new Map(entries.map((e: any) => [e.universe_id, e.rank] as [number, number])),
    });
  }
  reads.sort((a, b) => a.at - b.at);
  return reads.slice(-KEEP_READS);
}

function zeros(): Matrix {
  return Array.from({ length: STATES }, () => new Array<number>(STATES).fill(0));
}

function multiply(x: Matrix, y: Matrix): Matrix {
  const out = zeros();
  for (let i = 0; i < STATES; i++)
    for (let k = 0; k < STATES; k++)
      if (x[i][k] !== 0) for (let j = 0; j < STATES; j++) out[i][j] += x[i][k] * y[k][j];
  return out;
}

function power(m: Matrix, k: number): Matrix {
  let result = zeros();
  for (let i = 0; i < STATES; i++) result[i][i] = 1;
  for (let step = 0; step < Math.min(k, MAX_MULTIPLICATIONS); step++) result = multiply(result, m);
  return result;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function buildModel(reads: Read[]): { P: Matrix; medianGap: number } | null {
  const counts = zeros();
  const gaps: number[] = [];
  for (let i = 0; i < reads.length; i++) {
    for (let j = i + 1; j < reads.length; j++) {
      const dt = (reads[j].at - reads[i].at) / 3600000;
      if (dt < GAP_MIN_H || dt > GAP_MAX_H) continue;
      gaps.push(dt);
      for (let rank = 1; rank <= RANKS; rank++) {
        const game = reads[i].top[rank - 1];
        if (game === undefined) continue;
        const later = reads[j].rankOf.get(game);
        const col = later !== undefined && later >= 1 && later <= RANKS ? later - 1 : OUT;
        counts[rank][col] += 1;
      }
    }
  }
  if (gaps.length === 0) return null;
  const P = zeros();
  for (let rank = 1; rank <= RANKS; rank++) {
    const total = counts[rank].reduce((a, b) => a + b, 0);
    for (let col = 0; col < STATES; col++)
      P[rank][col] = (counts[rank][col] + ALPHA) / (total + ALPHA * STATES);
  }
  return { P, medianGap: median(gaps) };
}

function horizonMatrix(P: Matrix, medianGap: number, h: number): Matrix {
  const x = Math.min((h / medianGap) ** 0.3, 2.0);
  const floor = Math.floor(x);
  const frac = x - floor;
  const low = power(P, floor);
  const high = power(P, floor + 1);
  return low.map((row, i) => row.map((v, j) => (1 - frac) * v + frac * high[i][j]));
}

function baseline(top: number[]): Row[] {
  const rows: Row[] = [];
  const add = (rank: number, place: number, p: number) => {
    if (top[rank - 1] !== undefined) rows.push({ universe_id: top[rank - 1], place, p });
  };
  for (let place = 1; place <= PLACES; place++) {
    if (place === 1) {
      add(1, 1, 0.5);
      add(2, 1, 0.3);
    } else {
      add(place, place, 0.5);
      add(place - 1, place, 0.15);
      add(place + 1, place, 0.15);
    }
  }
  return rows;
}

function matrixMap(last: Read, sum: Matrix): Map<string, number> {
  const out = new Map<string, number>();
  for (let place = 1; place <= PLACES; place++) {
    const placeRows: Row[] = [];
    for (let rank = 1; rank <= RANKS; rank++) {
      const game = last.top[rank - 1];
      const p = sum[rank][place - 1];
      if (game !== undefined && p >= MIN_P) placeRows.push({ universe_id: game, place, p });
    }
    const total = placeRows.reduce((a, row) => a + row.p, 0);
    const scale = total > PLACE_CAP ? PLACE_CAP / total : 1;
    for (const row of placeRows) out.set(`${row.universe_id}:${place}`, row.p * scale);
  }
  return out;
}

function shiftMap(last: Read): Map<string, number> {
  const out = new Map<string, number>();
  const hedgeRanks: [number, number][] = [];
  for (let rank = HEDGE_FROM; rank <= Math.min(HEDGE_TO, last.top.length); rank++)
    hedgeRanks.push([rank, Math.exp(-(rank - HEDGE_FROM) / HEDGE_DECAY)]);
  const hedgeTotal = hedgeRanks.reduce((a, [, w]) => a + w, 0);
  const add = (key: string, p: number) => out.set(key, (out.get(key) ?? 0) + p);
  for (let place = 1; place <= PLACES; place++) {
    const mass = new Map<number, number>();
    for (const [shift, weight] of SHIFTS) {
      const rank = place + shift;
      const game = rank >= 1 ? last.top[rank - 1] : undefined;
      if (game !== undefined) mass.set(game, (mass.get(game) ?? 0) + weight);
    }
    const total = [...mass.values()].reduce((a, b) => a + b, 0);
    if (total > 0) for (const [game, m] of mass) add(`${game}:${place}`, (m / total) * (1 - HEDGE));
    for (const [rank, w] of hedgeRanks) {
      const game = last.top[rank - 1];
      if (game !== undefined) add(`${game}:${place}`, (HEDGE * w) / hedgeTotal);
    }
  }
  return out;
}

function forecast(reads: Read[], slots: number): { rows: Row[]; top10: number[] } {
  const last = reads[reads.length - 1];
  const top10 = last ? last.top.slice(0, PLACES) : [];
  const model = reads.length >= 2 ? buildModel(reads) : null;
  if (!model || !last) return { rows: last ? baseline(last.top) : [], top10 };

  const sum: Matrix = zeros();
  for (let h = 1; h <= slots; h++) {
    const A = horizonMatrix(model.P, model.medianGap, h);
    for (let i = 0; i < STATES; i++)
      for (let j = 0; j < STATES; j++) sum[i][j] += A[i][j] / slots;
  }

  const mapMatrix = matrixMap(last, sum);
  const mapShift = shiftMap(last);
  const rows: Row[] = [];
  for (const key of new Set([...mapMatrix.keys(), ...mapShift.keys()])) {
    const p = MATRIX_BLEND * (mapMatrix.get(key) ?? 0) + SHIFT_BLEND * (mapShift.get(key) ?? 0);
    if (p < MIN_P) continue;
    const [universe_id, place] = key.split(':').map(Number);
    rows.push({ universe_id, place, p });
  }
  rows.sort((a, b) => a.place - b.place || a.universe_id - b.universe_id);

  const placeTotal = new Map<number, number>();
  for (const row of rows) placeTotal.set(row.place, (placeTotal.get(row.place) ?? 0) + row.p);
  for (const row of rows) {
    const total = placeTotal.get(row.place) ?? 0;
    if (total > 1) row.p /= total;
  }
  return { rows, top10 };
}

const task = JSON.parse(readFileSync(join(inputDir, 'task.json'), 'utf8'));
const reads = loadReads(join(inputDir, 'history.ndjson'));
const slots = task.window?.slots ?? 24;
const { rows, top10 } = forecast(reads, slots);

mkdirSync(outputDir, { recursive: true });
const result = { probabilities: rows, top10 };
writeFileSync(join(outputDir, task.output ?? 'forecast.json'), JSON.stringify(result, null, 2) + '\n', 'utf8');
process.stderr.write(`forecast: ${rows.length} rows from ${reads.length} reads\n`);
