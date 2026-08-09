import Link from "next/link";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/matches", label: "Matches" },
  { href: "/calibration", label: "Calibration" },
  { href: "/market", label: "Market Benchmark" },
  { href: "/seasons", label: "Team Strength" },
];

export default function Nav() {
  return (
    <header className="border-b" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
      <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-4">
        <Link href="/" className="text-sm font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
          MatchState
        </Link>
        <nav className="flex gap-5 text-sm">
          {LINKS.map((l) => (
            <Link key={l.href} href={l.href} style={{ color: "var(--text-secondary)" }} className="hover:opacity-80">
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
