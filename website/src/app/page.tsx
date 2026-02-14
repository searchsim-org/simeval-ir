import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function Home() {
  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="text-center space-y-6 py-12">
        <Badge variant="outline" className="px-4 py-1">
          v0.1.0 — Now Available
        </Badge>
        <h1 className="text-5xl font-bold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
          SimEval-IR
        </h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          A unified framework for evaluating simulated search and conversational sessions.
          Behavioral realism, system effectiveness, and tester reliability — all in one place.
        </p>
        <div className="flex items-center justify-center gap-4">
          <Button size="lg" asChild>
            <Link href="/explorer">Explore Sessions</Link>
          </Button>
          <Button variant="outline" size="lg" asChild>
            <Link href="/docs">Read Docs</Link>
          </Button>
        </div>
      </section>

      {/* Features Grid */}
      <section className="grid md:grid-cols-3 gap-6">
        <FeatureCard
          title="Behavioral Realism"
          description="Measure how well simulators replicate real user behavior with JSD, Fréchet Distance, and classifier-based metrics."
          icon="🎯"
          metrics={["JSD", "FD", "Classifier AUC"]}
        />
        <FeatureCard
          title="System Effectiveness"
          description="Evaluate system performance with session-level nDCG, Expected Global Utility, and conversational metrics."
          icon="📊"
          metrics={["Session nDCG", "EGU", "Conv MRR"]}
        />
        <FeatureCard
          title="Tester Reliability"
          description="Assess how reliably simulators rank systems compared to real users using RATE and correlation metrics."
          icon="🔬"
          metrics={["Kendall τ", "RATE", "Spearman ρ"]}
        />
      </section>

      {/* Data Types Section */}
      <section className="space-y-6">
        <h2 className="text-2xl font-semibold text-center">Unified Data Model</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="border-indigo-500/30 bg-indigo-500/5">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span className="text-2xl">🔍</span>
                Traditional Search (T)
              </CardTitle>
              <CardDescription>
                Query-based sessions with rankings and clicks
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>• Query reformulation sequences</p>
                <p>• SERP views and click patterns</p>
                <p>• Dwell times and navigation</p>
                <p>• Relevance judgments when available</p>
              </div>
            </CardContent>
          </Card>
          <Card className="border-purple-500/30 bg-purple-500/5">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span className="text-2xl">💬</span>
                Conversational (C)
              </CardTitle>
              <CardDescription>
                Multi-turn dialogues with user and system utterances
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>• User and system utterances</p>
                <p>• Turn-based structure</p>
                <p>• Retrieval within conversations</p>
                <p>• Recommendations and feedback</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Integrations */}
      <section className="space-y-6">
        <h2 className="text-2xl font-semibold text-center">Integrations</h2>
        <div className="flex justify-center gap-8">
          <Card className="w-48 text-center">
            <CardHeader>
              <CardTitle className="text-lg">SimIIR</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Load logs from SimIIR simulations directly
              </p>
            </CardContent>
          </Card>
          <Card className="w-48 text-center">
            <CardHeader>
              <CardTitle className="text-lg">PyTerrier</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Convert runs and eval in IR pipelines
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Quick Install */}
      <section className="text-center space-y-4">
        <h2 className="text-2xl font-semibold">Get Started</h2>
        <div className="bg-card border rounded-lg p-4 max-w-md mx-auto font-mono text-sm">
          <span className="text-muted-foreground">$</span> pip install simeval-ir
        </div>
        <p className="text-muted-foreground">
          Dataset-agnostic. Extensible. Ready for your research.
        </p>
      </section>
    </div>
  );
}

function FeatureCard({
  title,
  description,
  icon,
  metrics,
}: {
  title: string;
  description: string;
  icon: string;
  metrics: string[];
}) {
  return (
    <Card className="hover:border-primary/50 transition-colors">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="text-2xl">{icon}</span>
          {title}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-2">
          {metrics.map((m) => (
            <Badge key={m} variant="secondary">
              {m}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
