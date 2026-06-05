/** @type {import('next').NextConfig} */
const nextConfig = {
  // 生产环境直连后端 (避免 Vercel 60s 超时影响 SSE 流式)
  // 开发环境通过环境变量 NEXT_PUBLIC_API_URL 配置
  images: { domains: [] },
};

module.exports = nextConfig;
