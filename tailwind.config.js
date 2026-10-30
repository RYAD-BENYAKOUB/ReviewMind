/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './web/**/*.{html,js,ts,jsx,tsx}',
    './templates/**/*.html',
    './static/**/*.{js,css}',
    './index.html',
    './src/**/*.{py,html}',
  ],
  theme: {
    extend: {
      colors: {
        // ReviewMind — Dark Intelligence Brand Palette (Definitive)
        brand: {
          black: '#0B0B0F', // Dominant ~75%
          surface: '#121217',
          card: '#16161C',
          border: '#24242F',
          blue: '#1E6BFF', // Secondary accent ~15%
          'blue-hover': '#1657D9',
          red: '#E0122F', // Complementary accent ~10%
          'red-hover': '#C00F27',
        },
        sentiment: {
          positive: '#1E6BFF',
          neutral: '#8E8E93',
          negative: '#E0122F',
        },
        text: {
          primary: '#FFFFFF',
          muted: '#8E8E93',
          dim: '#52525B',
        },
        border: {
          dark: '#24242F',
          subtle: '#1C1C24',
        },
      },
      fontFamily: {
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'Roboto',
          'sans-serif',
        ],
        mono: [
          'JetBrains Mono',
          'Fira Code',
          'SFMono-Regular',
          'Consolas',
          'monospace',
        ],
      },
    },
  },
  plugins: [],
};

