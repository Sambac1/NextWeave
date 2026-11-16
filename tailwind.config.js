// Research workspace. Retain paper/ink aliases for existing components.
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['"IBM Plex Sans"', '-apple-system', '"PingFang SC"', '"Microsoft YaHei"', 'sans-serif'],
        body: ['"IBM Plex Sans"', '-apple-system', '"PingFang SC"', '"Microsoft YaHei"', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'Menlo', 'monospace'],
      },
      colors: {
        paper: {
          50:  '#f9f9fc',
          100: '#f4f5fa',
          200: '#eaeaf3',
          300: '#dcdfea',
          400: '#c3c9da',
          500: '#909bb3',
          600: '#647b91',
          700: '#4d647a',
          800: '#344b63',
          900: '#172137',
          950: '#102236',
        },
        ink: {
          50:  '#f3f6fa',
          100: '#e8eef4',
          200: '#b9c8d6',
          300: '#607286',
          400: '#5a667d',
          500: '#4e5a73',
          600: '#424d66',
          700: '#343e57',
          800: '#252e46',
          900: '#172137',
        },
        brand: {
          50: '#f5f3ff', 100: '#ebe7ff', 200: '#d9d3fc',
          300: '#bcb2f5', 400: '#9c8eeb', 500: '#7566e3',
          600: '#5145d7', 700: '#493dc4', 800: '#3c309b', 900: '#312678',
        },
        signal: {
          red:    '#b42339',
          amber:  '#936017',
          green:  '#23714e',
          blue:   '#5145d7',
          purple: '#625b8e',
        },
        navy: {
          50:  '#e8edf2',
          100: '#c5d0db',
          200: '#9aafc2',
          300: '#6f8ea9',
          400: '#4a6d90',
          500: '#2e5277',
          600: '#1e3d5e',
          700: '#142d4a',
          800: '#0d1f2d',
          900: '#08131d',
          950: '#040a10',
        },
      },
      fontSize: {
        xs: ['0.8125rem', { lineHeight: '1.6' }],
        sm: ['0.9375rem', { lineHeight: '1.65' }],
        base: ['1rem', { lineHeight: '1.75' }],
      },
      backgroundImage: {
        'paper-texture': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 512 512' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.7' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E\")",
        'grain': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='5' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23g)' opacity='0.03'/%3E%3C/svg%3E\")",
        'scanline': "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.015) 2px, rgba(0,0,0,0.015) 4px)",
      },
      boxShadow: {
        'document': '0 2px 10px rgba(23,43,66,0.025)',
        'panel': '0 1px 3px rgba(23,43,66,0.035)',
        'elevated': '0 18px 60px rgba(23,43,66,0.14), 0 2px 8px rgba(23,43,66,0.05)',
        'stamp': '0 0 0 2px currentColor, inset 0 0 0 1px currentColor',
        'rule': '0 1px 0 0 rgba(0,0,0,0.08)',
      },
      animation: {
        'fade-in': 'fadeIn 0.16s ease-out',
        'slide-up': 'slideUp 0.18s cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-down': 'slideDown 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        'slide-right': 'slideRight 0.35s cubic-bezier(0.16, 1, 0.3, 1)',
        'reveal-right': 'revealRight 0.5s cubic-bezier(0.16, 1, 0.3, 1)',
        'typewriter': 'typewriter 1.5s steps(40)',
        'stamp-in': 'stampIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
        'marquee': 'marquee 30s linear infinite',
        'blink': 'blink 1s step-end infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideRight: {
          '0%': { opacity: '0', transform: 'translateX(-12px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        revealRight: {
          '0%': { clipPath: 'inset(0 100% 0 0)', opacity: '0' },
          '100%': { clipPath: 'inset(0 0 0 0)', opacity: '1' },
        },
        stampIn: {
          '0%': { transform: 'scale(1.4) rotate(-6deg)', opacity: '0' },
          '100%': { transform: 'scale(1) rotate(-3deg)', opacity: '0.85' },
        },
        marquee: {
          '0%': { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-50%)' },
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
      },
    },
  },
  plugins: [],
};
