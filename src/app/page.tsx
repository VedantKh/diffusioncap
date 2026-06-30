import FluidCanvas from "@/components/FluidCanvas";

export default function Home() {
  return (
    <main className="relative flex h-dvh w-full items-center justify-center overflow-hidden">
      <FluidCanvas />

      <div className="pointer-events-none relative z-10 flex select-none flex-col items-center px-6 text-center">
        <h1
          style={{ fontFamily: "var(--font-zodiak)" }}
          className="text-foreground text-[clamp(2.75rem,9vw,8rem)] font-bold leading-[0.95] tracking-tight"
        >
          diffusion capital
        </h1>
        <p className="text-foreground/40 mt-6 text-xs font-normal tracking-[0.25em] uppercase sm:text-sm">
          helping intelligence flow where it should
        </p>
      </div>
    </main>
  );
}
