// Roblox Top Trending forecast (make.fun roblox-trending-forecast-01).
//
// Model: a Markov chain over chart ranks. Each state is a rank 1..20 in an
// earlier read; the next state is that game's rank in a later read (1..20) or
// OUT (index 20, outside the top 20). Transitions come from every read pair
// whose gap is 0.4..16 hours, smoothed and row-normalised. The matrix for a
// horizon of h hours is P^x with x = min((h / medianGap)^0.3, 2.0), blended
// linearly between integer powers. The forecast averages those matrices over
// the 24 hourly horizons. Candidates are the games ranked 1..20 in the last
// read, scored for places 1..10; each place's probabilities are scaled so they
// sum to at most 0.85, leaving the rest as hedge for games not named.

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

  const rows: Row[] = [];
  for (let place = 1; place <= PLACES; place++) {
    const placeRows: Row[] = [];
    for (let rank = 1; rank <= RANKS; rank++) {
      const game = last.top[rank - 1];
      const p = sum[rank][place - 1];
      if (game !== undefined && p >= MIN_P) placeRows.push({ universe_id: game, place, p });
    }
    const total = placeRows.reduce((a, row) => a + row.p, 0);
    const scale = total > PLACE_CAP ? PLACE_CAP / total : 1;
    for (const row of placeRows) rows.push({ ...row, p: row.p * scale });
  }
  rows.sort((a, b) => a.place - b.place || a.universe_id - b.universe_id);
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
