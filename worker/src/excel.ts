/**
 * The admin Excel report, built straight from the database with ExcelJS.
 *
 * Untrusted text (prompts, responses, feedback, tester names) is always written
 * as a plain string cell, never as a formula, so a value such as
 * "=HYPERLINK(...)" is shown, not evaluated. Characters XML cannot hold are
 * dropped and cells are capped at Excel's 32,767-character limit.
 */
import ExcelJS from 'exceljs';
import { METRICS } from './evaluator';
import { aggregate, type Summary } from './reports';
import type { Row } from './store';
import { wallClock } from './time';

export const FILENAME = 'AI_Playground_Company_Testing_Logs.xlsx';
export const CELL_LIMIT = 32767;
const TRUNCATED = "… [truncated: exceeds Excel's cell limit]";
// eslint-disable-next-line no-control-regex
const ILLEGAL = /[\u0000-\u0008\u000B\u000C\u000E-\u001F￾￿]/g;

export function safeText(value: unknown): string | null {
  if (value == null) return null;
  let text = String(value).replace(ILLEGAL, '');
  if (text.length > CELL_LIMIT) text = text.slice(0, CELL_LIMIT - TRUNCATED.length) + TRUNCATED;
  return text;
}

const INT = '#,##0';
const COST = '0.000000';
const SCORE = '0.0';
const DATE = 'yyyy-mm-dd';
const TIME = 'hh:mm:ss';
const DATETIME = 'yyyy-mm-dd hh:mm:ss';
const TEXT = 'text';

type Column = { header: string; width: number; get: (r: any) => unknown; fmt?: string };
type Totals = Record<number, number | null>;

const solid = (argb: string): ExcelJS.Fill => ({ type: 'pattern', pattern: 'solid', fgColor: { argb } });
const HEADER_FILL = solid('FF1F2937');
const OK_FILL = solid('FFDCFCE7');
const FAIL_FILL = solid('FFFEE2E2');
const PENDING_FILL = solid('FFFEF3C7');

function addSheet(
  wb: ExcelJS.Workbook, title: string, subtitle: string, columns: Column[], rows: any[],
  opts: { status?: boolean; totals?: Totals } = {},
) {
  const ws = wb.addWorksheet(title);
  const headerRow = 4;
  ws.getCell(1, 1).value = title;
  ws.getCell(1, 1).font = { bold: true, size: 14 };
  ws.getCell(2, 1).value = safeText(subtitle);

  columns.forEach((c, i) => {
    const cell = ws.getCell(headerRow, i + 1);
    cell.value = c.header;
    cell.font = { bold: true, color: { argb: 'FFFFFFFF' } };
    cell.fill = HEADER_FILL;
    ws.getColumn(i + 1).width = c.width;
  });

  rows.forEach((row, ri) => {
    const r = headerRow + 1 + ri;
    columns.forEach((c, ci) => {
      const cell = ws.getCell(r, ci + 1);
      const raw = c.get(row);
      if (c.fmt === TEXT) {
        cell.value = safeText(raw);
        if (typeof raw === 'string' && raw.length > 60) cell.alignment = { wrapText: true, vertical: 'top' };
      } else {
        cell.value = raw as ExcelJS.CellValue;
        if (c.fmt) cell.numFmt = c.fmt;
      }
      if (opts.status && c.header === 'Status') {
        cell.fill = row.status === 'success' ? OK_FILL : row.status === 'failed' ? FAIL_FILL : PENDING_FILL;
      }
    });
  });

  const last = headerRow + rows.length;
  ws.views = [{ state: 'frozen', ySplit: headerRow }];
  ws.autoFilter = { from: { row: headerRow, column: 1 }, to: { row: Math.max(last, headerRow), column: columns.length } };

  if (opts.totals && rows.length) {
    const tr = last + 2;
    ws.getCell(tr, 1).value = 'Total';
    ws.getCell(tr, 1).font = { bold: true };
    for (const [colStr, total] of Object.entries(opts.totals)) {
      const col = Number(colStr);
      const letter = ws.getColumn(col).letter;
      // SUBTOTAL ignores rows hidden by the filter, so the total follows what the reader filters to.
      const cell = ws.getCell(tr, col);
      cell.value = { formula: `SUBTOTAL(109,${letter}${headerRow + 1}:${letter}${last})`, result: total ?? 0 };
      cell.font = { bold: true };
      cell.numFmt = columns[col - 1]!.fmt ?? INT;
    }
  }
  return ws;
}

const usd = (c: Record<string, number>) => c.USD ?? null;
const otherCurrencies = (c: Record<string, number>) =>
  Object.entries(c).filter(([k]) => k !== 'USD').map(([k, v]) => `${v.toFixed(6)} ${k}`).join(', ') || null;
const modeLabel = (m: string) => (m === 'kundli_chatbot' ? 'Kundli Chatbot' : 'Direct provider');

export async function buildWorkbook(rows: Row[], filterNote: string, tz: string): Promise<ArrayBuffer> {
  const agg = aggregate(rows, tz);
  const t = agg.totals;
  const sub = `Generated ${new Date().toISOString().slice(0, 16).replace('T', ' ')} UTC · times in ${tz} · ${filterNote}`;
  const wb = new ExcelJS.Workbook();
  wb.creator = 'AI Model Playground';

  const local = (epoch: number) => wallClock(epoch, tz);
  const timeOfDay = (epoch: number) => {
    const d = local(epoch);
    return (d.getUTCHours() * 3600 + d.getUTCMinutes() * 60 + d.getUTCSeconds()) / 86400;
  };
  const day = (epoch: number) => {
    const d = local(epoch);
    return new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
  };

  addSheet(wb, 'All Test Logs', `${sub} · first-token latency is only filled when the backend reports it`, [
    { header: 'Experiment ID', width: 13, get: (r) => r.id, fmt: INT },
    { header: 'Request ID', width: 26, get: (r) => r.request_id, fmt: TEXT },
    { header: 'Tester', width: 18, get: (r) => r.tester_name, fmt: TEXT },
    { header: 'Session ID', width: 38, get: (r) => r.session_id, fmt: TEXT },
    { header: 'Date', width: 12, get: (r) => day(r.created_at), fmt: DATE },
    { header: 'Time', width: 10, get: (r) => timeOfDay(r.created_at), fmt: TIME },
    { header: 'Testing mode', width: 16, get: (r) => modeLabel(r.mode), fmt: TEXT },
    { header: 'Provider', width: 12, get: (r) => r.provider, fmt: TEXT },
    { header: 'Model', width: 32, get: (r) => r.model, fmt: TEXT },
    { header: 'System prompt', width: 40, get: (r) => r.system_prompt, fmt: TEXT },
    { header: 'User prompt', width: 50, get: (r) => r.user_prompt, fmt: TEXT },
    { header: 'Context', width: 40, get: (r) => r.context, fmt: TEXT },
    { header: 'Generated response', width: 60, get: (r) => r.response, fmt: TEXT },
    { header: 'Original response', width: 60, get: (r) => r.original_response, fmt: TEXT },
    { header: 'Humanized response', width: 40, get: (r) => r.humanized_response, fmt: TEXT },
    { header: 'Input tokens', width: 12, get: (r) => r.input_tokens, fmt: INT },
    { header: 'Output tokens', width: 12, get: (r) => r.output_tokens, fmt: INT },
    { header: 'Total tokens', width: 12, get: (r) => r.total_tokens, fmt: INT },
    { header: 'Usage source', width: 13, get: (r) => r.usage_source, fmt: TEXT },
    { header: 'Input cost', width: 12, get: (r) => r.input_cost, fmt: COST },
    { header: 'Output cost', width: 12, get: (r) => r.output_cost, fmt: COST },
    { header: 'Total cost', width: 12, get: (r) => r.total_cost, fmt: COST },
    { header: 'Currency', width: 9, get: (r) => r.currency, fmt: TEXT },
    { header: 'Latency (ms)', width: 12, get: (r) => r.latency_ms, fmt: INT },
    { header: 'First token latency (ms)', width: 14, get: (r) => r.first_token_latency_ms, fmt: INT },
    { header: 'Status', width: 10, get: (r) => r.status, fmt: TEXT },
    { header: 'Error', width: 40, get: (r) => r.error, fmt: TEXT },
  ], rows, { status: true, totals: { 16: t.input_tokens, 17: t.output_tokens, 18: t.total_tokens, 22: usd(t.cost) } });

  const sumTotals = (extra: Totals = {}): Totals => ({ ...extra });
  addSheet(wb, 'Model Summary', sub, [
    { header: 'Provider', width: 14, get: (r) => r.provider, fmt: TEXT },
    { header: 'Model', width: 36, get: (r) => r.model, fmt: TEXT },
    { header: 'Total requests', width: 14, get: (r) => r.requests, fmt: INT },
    { header: 'Successful', width: 12, get: (r) => r.success, fmt: INT },
    { header: 'Failed', width: 10, get: (r) => r.failed, fmt: INT },
    { header: 'Input tokens', width: 14, get: (r) => r.input_tokens, fmt: INT },
    { header: 'Output tokens', width: 14, get: (r) => r.output_tokens, fmt: INT },
    { header: 'Total tokens', width: 14, get: (r) => r.total_tokens, fmt: INT },
    { header: 'Total cost (USD)', width: 16, get: (r) => usd(r.cost), fmt: COST },
    { header: 'Other currencies', width: 18, get: (r) => otherCurrencies(r.cost), fmt: TEXT },
    { header: 'Avg latency (ms)', width: 16, get: (r) => r.avg_latency_ms, fmt: INT },
    { header: 'Avg AI score', width: 13, get: (r) => r.avg_score, fmt: SCORE },
  ], agg.by_model, { totals: sumTotals({ 3: t.requests, 4: t.success, 5: t.failed, 6: t.input_tokens, 7: t.output_tokens, 8: t.total_tokens, 9: usd(t.cost) }) });

  addSheet(wb, 'Token Analysis', sub, [
    { header: 'Date', width: 12, get: (r) => r.date, fmt: TEXT },
    { header: 'Provider', width: 14, get: (r) => r.provider, fmt: TEXT },
    { header: 'Model', width: 36, get: (r) => r.model, fmt: TEXT },
    { header: 'Requests', width: 10, get: (r) => r.requests, fmt: INT },
    { header: 'Input tokens', width: 14, get: (r) => r.input_tokens, fmt: INT },
    { header: 'Output tokens', width: 14, get: (r) => r.output_tokens, fmt: INT },
    { header: 'Total tokens', width: 14, get: (r) => r.total_tokens, fmt: INT },
    { header: 'Avg tokens / request', width: 18, get: (r) => (r.requests ? Math.round(r.total_tokens / r.requests) : null), fmt: INT },
  ], agg.daily_model, { totals: { 4: t.requests, 5: t.input_tokens, 6: t.output_tokens, 7: t.total_tokens } });

  const costRows = agg.daily_model.flatMap((r) => Object.keys(r.cost).sort().map((currency) => ({
    ...r, currency, in: r.input_cost[currency] ?? null, out: r.output_cost[currency] ?? null, tot: r.cost[currency] ?? null,
  })));
  addSheet(wb, 'Cost Analysis', `${sub} · one row per currency; currencies are never added together`, [
    { header: 'Date', width: 12, get: (r) => r.date, fmt: TEXT },
    { header: 'Provider', width: 14, get: (r) => r.provider, fmt: TEXT },
    { header: 'Model', width: 36, get: (r) => r.model, fmt: TEXT },
    { header: 'Input cost', width: 14, get: (r) => r.in, fmt: COST },
    { header: 'Output cost', width: 14, get: (r) => r.out, fmt: COST },
    { header: 'Total estimated cost', width: 18, get: (r) => r.tot, fmt: COST },
    { header: 'Currency', width: 10, get: (r) => r.currency, fmt: TEXT },
  ], costRows);

  addSheet(wb, 'Daily Statistics', sub, [
    { header: 'Date', width: 12, get: (r) => r.date, fmt: TEXT },
    { header: 'Total experiments', width: 16, get: (r) => r.requests, fmt: INT },
    { header: 'Successful', width: 12, get: (r) => r.success, fmt: INT },
    { header: 'Failed', width: 10, get: (r) => r.failed, fmt: INT },
    { header: 'Unique sessions', width: 15, get: (r) => r.unique_sessions, fmt: INT },
    { header: 'Total tokens', width: 14, get: (r) => r.total_tokens, fmt: INT },
    { header: 'Total cost (USD)', width: 16, get: (r) => usd(r.cost), fmt: COST },
    { header: 'Other currencies', width: 18, get: (r) => otherCurrencies(r.cost), fmt: TEXT },
  ], agg.daily, { totals: { 2: t.requests, 3: t.success, 4: t.failed, 6: t.total_tokens, 7: usd(t.cost) } });

  addSheet(wb, 'Failed Requests', sub, [
    { header: 'Experiment ID', width: 13, get: (r) => r.id, fmt: INT },
    { header: 'Timestamp', width: 20, get: (r) => local(r.created_at), fmt: DATETIME },
    { header: 'Tester', width: 18, get: (r) => r.tester_name, fmt: TEXT },
    { header: 'Session ID', width: 38, get: (r) => r.session_id, fmt: TEXT },
    { header: 'Testing mode', width: 16, get: (r) => modeLabel(r.mode), fmt: TEXT },
    { header: 'Provider', width: 12, get: (r) => r.provider, fmt: TEXT },
    { header: 'Model', width: 30, get: (r) => r.model, fmt: TEXT },
    { header: 'Prompt', width: 60, get: (r) => r.user_prompt, fmt: TEXT },
    { header: 'Error details', width: 60, get: (r) => r.error, fmt: TEXT },
    { header: 'Status', width: 10, get: (r) => r.status, fmt: TEXT },
  ], rows.filter((r) => r.status === 'failed'), { status: true });

  const scores = (r: Row) => r.evaluation?.scores ?? {};
  addSheet(wb, 'Feedback & Evaluation', `${sub} · AI scores are an automated estimate, not ground truth`, [
    { header: 'Experiment ID', width: 13, get: (r) => r.id, fmt: INT },
    { header: 'Tester', width: 18, get: (r) => r.tester_name, fmt: TEXT },
    { header: 'Session ID', width: 38, get: (r) => r.session_id, fmt: TEXT },
    { header: 'Testing mode', width: 16, get: (r) => modeLabel(r.mode), fmt: TEXT },
    { header: 'Provider', width: 12, get: (r) => r.provider, fmt: TEXT },
    { header: 'Model', width: 30, get: (r) => r.model, fmt: TEXT },
    ...METRICS.map((m): Column => ({
      header: m.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase()), width: 13, get: (r) => scores(r)[m] ?? null, fmt: SCORE,
    })),
    { header: 'AI overall score', width: 14, get: (r) => r.evaluation_overall, fmt: SCORE },
    { header: 'AI feedback', width: 50, get: (r) => r.evaluation?.feedback, fmt: TEXT },
    { header: 'Evaluator model', width: 28, get: (r) => r.evaluation?.evaluator?.model, fmt: TEXT },
    { header: 'Human rating', width: 12, get: (r) => r.human_rating, fmt: TEXT },
    { header: 'Human feedback', width: 50, get: (r) => r.human_feedback, fmt: TEXT },
    { header: 'Edited response', width: 60, get: (r) => r.edited_response, fmt: TEXT },
  ], rows.filter((r) => r.evaluation || r.human_rating || r.human_feedback || r.edited_response));

  const sessionCols = (s: any) => ({ ...s });
  addSheet(wb, 'Session Summary', sub, [
    { header: 'Session ID', width: 38, get: (r) => r.session_id, fmt: TEXT },
    { header: 'Tester', width: 18, get: (r) => r.tester_name, fmt: TEXT },
    { header: 'Total experiments', width: 16, get: (r) => r.requests, fmt: INT },
    { header: 'Models tested', width: 50, get: (r) => r.models.join(', '), fmt: TEXT },
    { header: 'Total tokens', width: 14, get: (r) => r.total_tokens, fmt: INT },
    { header: 'Total cost (USD)', width: 16, get: (r) => usd(r.cost), fmt: COST },
    { header: 'First activity', width: 20, get: (r) => local(r.first), fmt: DATETIME },
    { header: 'Last activity', width: 20, get: (r) => local(r.last), fmt: DATETIME },
  ], agg.sessions.map(sessionCols));

  const qualityCols: Column[] = METRICS.map((m) => ({
    header: `Avg ${m.replace(/_/g, ' ')}`, width: 14, get: (r: Summary) => r.quality[m] ?? null, fmt: SCORE,
  }));
  addSheet(wb, 'Tester Summary', `${sub} · quality scores are averages of automated AI evaluations`, [
    { header: 'Tester', width: 22, get: (r) => r.tester, fmt: TEXT },
    { header: 'Sessions', width: 10, get: (r) => r.unique_sessions, fmt: INT },
    { header: 'Total experiments', width: 16, get: (r) => r.requests, fmt: INT },
    { header: 'Successful', width: 12, get: (r) => r.success, fmt: INT },
    { header: 'Failed', width: 10, get: (r) => r.failed, fmt: INT },
    { header: 'Input tokens', width: 14, get: (r) => r.input_tokens, fmt: INT },
    { header: 'Output tokens', width: 14, get: (r) => r.output_tokens, fmt: INT },
    { header: 'Total tokens', width: 14, get: (r) => r.total_tokens, fmt: INT },
    { header: 'Total cost (USD)', width: 16, get: (r) => usd(r.cost), fmt: COST },
    { header: 'Other currencies', width: 18, get: (r) => otherCurrencies(r.cost), fmt: TEXT },
    { header: 'Avg latency (ms)', width: 16, get: (r) => r.avg_latency_ms, fmt: INT },
    { header: 'Avg AI score', width: 13, get: (r) => r.avg_score, fmt: SCORE },
    { header: 'Good ratings', width: 12, get: (r) => r.good, fmt: INT },
    { header: 'Bad ratings', width: 12, get: (r) => r.bad, fmt: INT },
    ...qualityCols,
  ], agg.by_tester, { totals: { 3: t.requests, 4: t.success, 5: t.failed, 6: t.input_tokens, 7: t.output_tokens, 8: t.total_tokens, 9: usd(t.cost) } });

  return (await wb.xlsx.writeBuffer()) as ArrayBuffer;
}
