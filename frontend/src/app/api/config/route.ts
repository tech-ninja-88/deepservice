/**
 * 返回运行时配置 — 前端启动时调用此接口获取后端 API 地址
 * 这样更换后端地址时无需重新构建前端镜像
 */
export async function GET() {
  const apiUrl = process.env.API_URL || process.env.NEXT_PUBLIC_API_URL || "";
  return Response.json({ apiUrl });
}
