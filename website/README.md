# SimEval-IR Website Demo

Interactive explorer for SimEval-IR — visualize and evaluate simulated search and conversational sessions.

## Features

- **Session Explorer**: Interactive timeline visualization of search and conversational sessions
- **Metric Selector**: Choose from behavioral, system, and tester metrics
- **Real vs Simulated Comparison**: Side-by-side statistics
- **Metrics Reference**: Full documentation of all available metrics

## Tech Stack

- **Next.js 15** with App Router
- **TypeScript**
- **Tailwind CSS v4**
- **shadcn/ui** components

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Project Structure

```
src/
├── app/
│   ├── layout.tsx        # Root layout with navbar
│   ├── page.tsx          # Landing page
│   ├── explorer/         # Session explorer
│   └── metrics/          # Metrics documentation
├── components/
│   ├── ui/               # shadcn/ui components
│   ├── layout/           # Navbar, footer
│   └── explorer/         # Session viewer, metric selector
└── lib/
    ├── utils.ts          # Utility functions
    └── sample-data.ts    # Demo session data
```

## Pages

- `/` — Landing page with overview
- `/explorer` — Interactive session explorer
- `/metrics` — Metrics reference documentation
- `/datasets` — Dataset browser (coming soon)
- `/docs` — API documentation (coming soon)

## License

MIT
