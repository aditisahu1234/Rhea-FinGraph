import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rhea FinGraph",
  description: "Defense-only merchant fraud intelligence",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
