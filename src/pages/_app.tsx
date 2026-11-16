
import "@/styles/globals.css";
import Head from "next/head";
import type { AppProps } from "next/app";
import { Toaster } from "react-hot-toast";

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>NextWeave | Event Forecasting</title>
        <meta name="description" content="Forecast an event from a question or Polymarket contract using Wikipedia evidence and simulated decisions." />
        <meta property="og:site_name" content="NextWeave" />
        <link rel="icon" type="image/png" href="/brand/nextweave-icon.png" sizes="120x120" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
      <Component {...pageProps} />
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: "#ffffff",
            backdropFilter: "blur(12px)",
            WebkitBackdropFilter: "blur(12px)",
            color: "var(--workspace-text)",
            border: "1px solid var(--workspace-line)",
            fontSize: "14px",
            borderRadius: "10px",
            boxShadow: "0 8px 28px rgba(23, 43, 66, 0.1)",
          },
          success: {
            iconTheme: { primary: "#23714e", secondary: "#ffffff" },
          },
          error: {
            iconTheme: { primary: "#b42339", secondary: "#ffffff" },
            duration: 6000,
          },
        }}
      />
    </>
  );
}
