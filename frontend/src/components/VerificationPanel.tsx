import type { ClaimStatus, Experiment } from '../types';

const STATUS_LABEL: Record<ClaimStatus, string> = {
  SUPPORTED: 'Supported',
  CONTRADICTED: 'Contradicted',
  UNVERIFIABLE: 'Unverifiable',
  NOT_CHART_FACT: 'Not a chart fact',
};
const STATUS_CLASS: Record<ClaimStatus, string> = {
  SUPPORTED: 'tone-good',
  CONTRADICTED: 'tone-bad',
  UNVERIFIABLE: 'tone-neutral',
  NOT_CHART_FACT: 'tone-neutral',
};

interface Props {
  experiment: Experiment | null;
}

export default function VerificationPanel({ experiment }: Props) {
  if (!experiment || experiment.mode !== 'kundli_chatbot' || experiment.status !== 'success') return null;
  const v = experiment.verification;

  return (
    <section className="section">
      <h2 className="section-title">Response verification</h2>
      <p className="hint">
        Checks every chart fact the reply claims against the reference Kundli. Never proof that the whole reply
        (predictions, remedies, interpretation) is correct — only that its chart facts match.
      </p>
      {!v && <p className="empty">Generate a reference Kundli to verify this reply's chart claims.</p>}
      {v && (
        <>
          <div className="scores">
            <div className="score-row"><span>Supported</span><span /><span className="mono score-num tone-good">{v.supported_count}</span></div>
            <div className="score-row"><span>Contradicted</span><span /><span className="mono score-num tone-bad">{v.contradicted_count}</span></div>
            <div className="score-row"><span>Unverifiable</span><span /><span className="mono score-num">{v.unverifiable_count}</span></div>
            <div className="score-row overall">
              <span>Chart-fact consistency</span><span />
              <span className="mono score-num">
                {v.consistency_percent == null ? 'Insufficient evidence' : `${v.consistency_percent}%`}
              </span>
            </div>
          </div>

          {v.claims.length === 0
            ? <p className="empty">The reply made no chart-fact claims to check.</p>
            : (
              <div className="list-block">
                <h3 className="sub-title">Claims</h3>
                <ul className="claim-list">
                  {v.claims.map((c, i) => (
                    <li key={i} className="claim-row">
                      <span className={`chip ${STATUS_CLASS[c.status]}`}>{STATUS_LABEL[c.status]}</span>
                      <div>
                        <p className="claim-text">{c.claim}</p>
                        <p className="hint mono">
                          {c.attribute}: reply says "{c.statedValue}"
                          {c.referenceValue != null && ` · chart says ${c.referenceValue}`}
                        </p>
                        <p className="hint">{c.explanation}</p>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          <p className="hint mono">Claim extraction: {v.evaluator.model} (the pass/fail comparison itself is not AI-judged).</p>
        </>
      )}
    </section>
  );
}
