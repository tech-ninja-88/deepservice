/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  images: { domains: [] },

  // Proxy /api/* to backend in production (Sealos internal service discovery).
  // Browser calls same origin → no CORS needed.
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://backend:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
