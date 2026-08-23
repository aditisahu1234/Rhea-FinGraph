const foundations = [
  ["Data contract", "Canonical payment event schema is ready."],
  ["Risk API", "FastAPI is ready for the first trained model."],
  ["Graph stores", "Neo4j, PostgreSQL, Redis, and Elasticsearch are containerized."],
  ["Resilience", "Helix is isolated locally with telemetry disabled."],
];

export default function Home() {
  return (
    <main>
      <section className="hero">
        <p className="eyebrow">DAY 1 · FOUNDATION</p>
        <h1>Rhea FinGraph</h1>
        <p className="lede">
          Temporal graph intelligence for defensible merchant fraud decisions.
        </p>
        <span className="badge">Defense-only · model not trained yet</span>
      </section>
      <section className="grid" aria-label="Foundation status">
        {foundations.map(([title, detail]) => (
          <article className="card" key={title}>
            <p className="status">READY</p>
            <h2>{title}</h2>
            <p>{detail}</p>
          </article>
        ))}
      </section>
      <section className="next">
        <p className="eyebrow">NEXT MILESTONE</p>
        <h2>Profile IBM&apos;s fraud data and create a leakage-safe temporal split.</h2>
      </section>
    </main>
  );
}
