"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";

export function Navbar() {
    return (
        <nav className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
            <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                <Link href="/" className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                        <span className="text-white font-bold text-sm">SE</span>
                    </div>
                    <span className="font-semibold text-lg">SimEval-IR</span>
                </Link>

                <div className="flex items-center gap-6">
                    <Link href="/explorer" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                        Explorer
                    </Link>
                    <Link href="/metrics" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                        Metrics
                    </Link>
                    <Link href="/datasets" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                        Datasets
                    </Link>
                    <Link href="/docs" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                        Docs
                    </Link>
                    <Button variant="outline" size="sm" asChild>
                        <a href="https://github.com/simeval-ir/simeval-ir" target="_blank" rel="noopener noreferrer">
                            GitHub
                        </a>
                    </Button>
                </div>
            </div>
        </nav>
    );
}
