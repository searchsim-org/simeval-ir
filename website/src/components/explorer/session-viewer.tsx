"use client";

import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Session {
    session_id: string;
    session_type: "search" | "conversational";
    events: Event[];
}

interface Event {
    event_id: string;
    type: string;
    timestamp?: number;
    query?: string;
    utterance?: string;
    role?: "user" | "system";
    clicked_items?: { doc_id: string; rank: number }[];
}

interface SessionViewerProps {
    session: Session;
}

export function SessionViewer({ session }: SessionViewerProps) {
    return (
        <ScrollArea className="h-[400px]">
            <div className="space-y-4">
                {session.events.map((event, idx) => (
                    <EventCard key={event.event_id} event={event} index={idx} />
                ))}
            </div>
        </ScrollArea>
    );
}

function EventCard({ event, index }: { event: Event; index: number }) {
    const isUser = event.role === "user" ||
        event.type === "query_issued" ||
        event.type === "click" ||
        event.type === "user_utterance";

    const getEventIcon = (type: string) => {
        switch (type) {
            case "query_issued": return "🔍";
            case "serp_view": return "📋";
            case "click": return "👆";
            case "user_utterance": return "👤";
            case "system_utterance": return "🤖";
            default: return "📌";
        }
    };

    const getEventColor = (type: string) => {
        switch (type) {
            case "query_issued": return "bg-blue-500/10 border-blue-500/30";
            case "serp_view": return "bg-gray-500/10 border-gray-500/30";
            case "click": return "bg-green-500/10 border-green-500/30";
            case "user_utterance": return "bg-indigo-500/10 border-indigo-500/30";
            case "system_utterance": return "bg-purple-500/10 border-purple-500/30";
            default: return "bg-muted/50 border-muted";
        }
    };

    return (
        <div className={`flex gap-4 ${isUser ? "" : "pl-8"}`}>
            <div className="flex flex-col items-center">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg border ${getEventColor(event.type)}`}>
                    {getEventIcon(event.type)}
                </div>
                {index < 10 && <div className="w-0.5 h-4 bg-border" />}
            </div>

            <div className={`flex-1 p-3 rounded-lg border ${getEventColor(event.type)}`}>
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">
                            {event.type.replace(/_/g, " ")}
                        </Badge>
                        {event.timestamp !== undefined && (
                            <span className="text-xs text-muted-foreground">
                                {event.timestamp.toFixed(1)}s
                            </span>
                        )}
                    </div>
                </div>

                {event.query && (
                    <p className="font-medium">&quot;{event.query}&quot;</p>
                )}

                {event.utterance && (
                    <p className={isUser ? "font-medium" : "text-muted-foreground"}>
                        {event.utterance}
                    </p>
                )}

                {event.clicked_items && event.clicked_items.length > 0 && (
                    <div className="flex gap-2 mt-2">
                        {event.clicked_items.map((item) => (
                            <Badge key={item.doc_id} variant="secondary" className="text-xs">
                                {item.doc_id} (rank {item.rank})
                            </Badge>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
