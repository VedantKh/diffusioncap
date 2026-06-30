import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import localFont from "next/font/local";
import "./globals.css";

const jakarta = Plus_Jakarta_Sans({
  variable: "--font-jakarta",
  subsets: ["latin"],
});

const zodiak = localFont({
  src: "./fonts/Zodiak-Bold.woff2",
  weight: "700",
  style: "normal",
  variable: "--font-zodiak",
  display: "swap",
});

export const metadata: Metadata = {
  title: "diffusion capital",
  description: "diffusion capital",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${jakarta.variable} ${zodiak.variable} h-full antialiased`}
    >
      <body className="min-h-full">{children}</body>
    </html>
  );
}
