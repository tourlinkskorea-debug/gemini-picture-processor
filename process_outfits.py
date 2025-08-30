#!/usr/bin/env python3
"""
Outfit图片处理脚本
使用LLM处理Outfit文件夹中的图片，生成模特拍摄图
"""

import os
import base64
import json
import time
import asyncio
import aiofiles
from datetime import datetime
from pathlib import Path

try:
    from openai import AsyncOpenAI
    import httpx
except ImportError:
    print("错误: 未安装必要的库，请运行: pip install openai httpx aiofiles")
    exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("错误: 未安装 python-dotenv 库，请运行: pip install python-dotenv")
    exit(1)

try:
    from config import PROMPT_TEXT
except ImportError:
    print("错误: 无法导入config.py中的配置，请检查config.py文件")
    exit(1)

# 加载环境变量
load_dotenv()

class OutfitProcessor:
    def __init__(self):
        # 初始化OpenRouter API客户端
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY 环境变量未设置，请检查.env文件")
        
        # 创建异步HTTP客户端
        self.http_client = httpx.AsyncClient()
        
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            http_client=self.http_client,
        )
        
        # 设置文件夹路径
        self.outfit_dir = Path("Outfit")
        self.output_dir = Path("Output")
        
        # 确保输出文件夹存在
        self.output_dir.mkdir(exist_ok=True)
        
        # 生成本次处理的时间戳
        self.timestamp = int(time.time())
        
        # 支持的图片格式
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        # 重试配置
        self.max_retry_attempts = 3  # 最大重试次数
        self.retry_delay = 5  # 重试延迟（秒）
        self.retry_delay_multiplier = 1.5  # 重试延迟倍数（指数退避）
        
        # 并发控制
        self.max_concurrent_requests = 100  # 最大并发请求数
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        print(f"初始化完成，时间戳: {self.timestamp}")
        print(f"重试配置: 最大重试次数={self.max_retry_attempts}, 基础延迟={self.retry_delay}秒")
        print(f"并发配置: 最大并发请求数={self.max_concurrent_requests}")
    
    async def encode_image_to_base64(self, image_path):
        """将图片文件编码为base64格式"""
        try:
            async with aiofiles.open(image_path, "rb") as image_file:
                content = await image_file.read()
                encoded_string = base64.b64encode(content).decode('utf-8')
                # 获取图片格式
                ext = image_path.suffix.lower()
                if ext == '.jpg':
                    ext = '.jpeg'
                mime_type = f"image/{ext[1:]}"
                return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            print(f"编码图片失败 {image_path}: {e}")
            return None
    
    def get_images_from_folder(self, folder_path):
        """获取文件夹中的所有图片文件"""
        images = []
        try:
            for file_path in folder_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                    images.append(file_path)
            return sorted(images)  # 排序以确保处理顺序一致
        except Exception as e:
            print(f"读取文件夹失败 {folder_path}: {e}")
            return []
    
    async def call_llm_api(self, image_urls, folder_name):
        """调用LLM API处理图片，带重试机制"""
        # 构建消息内容
        content = [
            {
                "type": "text",
                "text": PROMPT_TEXT
            }
        ]
        
        # 添加所有图片
        for url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": url}
            })
        
        print(f"正在处理 {folder_name}，图片数量: {len(image_urls)}")
        
        # 重试统计
        retry_info = {
            "total_attempts": 0,
            "successful": False,
            "final_attempt": 0,
            "had_retries": False
        }
        
        # 重试循环
        for attempt in range(self.max_retry_attempts + 1):  # +1 包含首次尝试
            retry_info["total_attempts"] += 1
            retry_info["final_attempt"] = attempt + 1
            
            try:
                if attempt > 0:
                    retry_info["had_retries"] = True
                    # 计算重试延迟（指数退避）
                    delay = self.retry_delay * (self.retry_delay_multiplier ** (attempt - 1))
                    print(f"  第 {attempt} 次重试，延迟 {delay:.1f} 秒...")
                    await asyncio.sleep(delay)
                
                # 调用API
                completion = await self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": os.getenv("SITE_URL", ""),
                        "X-Title": os.getenv("SITE_NAME", ""),
                    },
                    model="google/gemini-2.5-flash-image-preview",
                    messages=[{
                        "role": "user",
                        "content": content
                    }]
                )
                
                # 检查响应是否包含图片
                if self.has_images_in_response(completion):
                    retry_info["successful"] = True
                    if attempt > 0:
                        print(f"  重试成功！在第 {attempt + 1} 次尝试中获得图片")
                    return completion, retry_info
                else:
                    if attempt < self.max_retry_attempts:
                        print(f"  第 {attempt + 1} 次尝试未返回图片，准备重试...")
                    else:
                        print(f"  所有重试均未返回图片，保存文本响应用于调试")
                        return completion, retry_info  # 返回最后一次响应，即使没有图片
                
            except Exception as e:
                if attempt < self.max_retry_attempts:
                    print(f"  第 {attempt + 1} 次API调用失败: {e}，准备重试...")
                else:
                    print(f"  所有重试均失败，最后错误: {e}")
                    return None, retry_info
        
        return None, retry_info
    
    def has_images_in_response(self, response):
        """检查响应中是否包含图片"""
        try:
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                
                # 检查message.images
                if hasattr(message, 'images') and message.images:
                    return True
                
                # 检查content中的base64图片
                if hasattr(message, 'content') and isinstance(message.content, str):
                    import re
                    base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                    matches = re.findall(base64_pattern, message.content)
                    if matches:
                        return True
                        
            return False
        except Exception as e:
            print(f"检查响应图片时出错: {e}")
            return False
    
    async def save_generated_images(self, response, folder_name):
        """保存生成的图片"""
        saved_files = []
        
        try:
            # 检查响应格式
            if hasattr(response, 'choices') and response.choices:
                message = response.choices[0].message
                
                # 处理可能的图片返回格式
                images = []
                
                # 格式1: message.images (如参考材料所示)
                if hasattr(message, 'images') and message.images:
                    images = message.images
                
                # 格式2: 检查content中是否包含图片
                elif hasattr(message, 'content'):
                    # 如果content是字符串，可能包含base64图片数据
                    if isinstance(message.content, str):
                        # 查找base64图片数据
                        import re
                        base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
                        matches = re.findall(base64_pattern, message.content)
                        for i, match in enumerate(matches):
                            images.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{match}"}
                            })
                
                # 保存图片
                for i, image in enumerate(images):
                    try:
                        image_url = image["image_url"]["url"]
                        
                        # 提取base64数据
                        if "base64," in image_url:
                            base64_data = image_url.split("base64,")[1]
                            
                            # 生成文件名
                            filename = f"{folder_name}_{self.timestamp}_{i+1}.png"
                            filepath = self.output_dir / filename
                            
                            # 保存图片
                            async with aiofiles.open(filepath, "wb") as f:
                                await f.write(base64.b64decode(base64_data))
                            
                            saved_files.append(filepath)
                            print(f"保存图片: {filename}")
                            
                    except Exception as e:
                        print(f"保存图片失败 {i+1}: {e}")
                
                # 如果没有找到图片，保存文本响应用于调试
                if not images and hasattr(message, 'content'):
                    debug_file = self.output_dir / f"{folder_name}_{self.timestamp}_response.txt"
                    async with aiofiles.open(debug_file, "w", encoding="utf-8") as f:
                        await f.write(str(message.content))
                    print(f"保存调试响应: {debug_file}")
                        
        except Exception as e:
            print(f"保存图片过程中出错: {e}")
        
        return saved_files
    
    async def close(self):
        """关闭异步客户端"""
        await self.http_client.aclose()
    
    async def process_single_folder(self, folder):
        """处理单个文件夹"""
        async with self.semaphore:  # 并发控制
            print(f"\n开始处理文件夹: {folder.name}")
            
            # 获取文件夹中的图片
            images = self.get_images_from_folder(folder)
            
            if not images:
                error_msg = f"文件夹 {folder.name} 中没有图片文件"
                print(error_msg)
                return {
                    "folder_name": folder.name,
                    "success": False,
                    "error": error_msg,
                    "retry_info": {"total_attempts": 0, "successful": False}
                }
            
            # 将图片编码为base64
            image_urls = []
            for img_path in images:
                encoded = await self.encode_image_to_base64(img_path)
                if encoded:
                    image_urls.append(encoded)
                    print(f"  编码图片: {img_path.name}")
            
            if not image_urls:
                error_msg = f"文件夹 {folder.name} 中没有可用的图片"
                print(error_msg)
                return {
                    "folder_name": folder.name,
                    "success": False,
                    "error": error_msg,
                    "retry_info": {"total_attempts": 0, "successful": False}
                }
            
            # 调用LLM API
            api_result = await self.call_llm_api(image_urls, folder.name)
            
            if api_result and len(api_result) == 2:
                response, retry_info = api_result
                
                if response:
                    # 保存生成的图片
                    saved_files = await self.save_generated_images(response, folder.name)
                    
                    result = {
                        "folder_name": folder.name,
                        "input_images": [img.name for img in images],
                        "generated_images": [f.name for f in saved_files],
                        "success": True,
                        "retry_info": retry_info
                    }
                    
                    if saved_files:
                        print(f"文件夹 {folder.name} 处理完成，生成 {len(saved_files)} 张图片")
                    else:
                        print(f"文件夹 {folder.name} 处理完成，但未生成图片（已保存调试信息）")
                    
                    return result
                else:
                    error_msg = f"处理文件夹 {folder.name} 时API响应为空"
                    print(error_msg)
                    return {
                        "folder_name": folder.name,
                        "success": False,
                        "error": error_msg,
                        "retry_info": retry_info
                    }
            else:
                error_msg = f"处理文件夹 {folder.name} 时API调用失败"
                print(error_msg)
                return {
                    "folder_name": folder.name,
                    "success": False,
                    "error": error_msg,
                    "retry_info": {"total_attempts": 0, "successful": False}
                }
    
    async def process_all_folders(self):
        """并发处理所有outfit子文件夹"""
        results = {
            "timestamp": self.timestamp,
            "processed_folders": [],
            "total_images_generated": 0,
            "errors": [],
            "retry_statistics": {
                "total_api_calls": 0,
                "successful_retries": 0,
                "failed_after_retries": 0
            }
        }
        
        try:
            # 获取所有子文件夹
            subfolders = [f for f in self.outfit_dir.iterdir() if f.is_dir()]
            
            if not subfolders:
                print("未找到任何子文件夹")
                return results
            
            print(f"找到 {len(subfolders)} 个子文件夹，开始并发处理...")
            print(f"最大并发数: {self.max_concurrent_requests}")
            
            # 并发处理所有文件夹
            tasks = [self.process_single_folder(folder) for folder in subfolders]
            folder_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for i, folder_result in enumerate(folder_results):
                if isinstance(folder_result, Exception):
                    error_msg = f"处理文件夹 {subfolders[i].name} 时发生异常: {folder_result}"
                    results["errors"].append(error_msg)
                    print(error_msg)
                elif folder_result and isinstance(folder_result, dict):
                    # 更新重试统计
                    if "retry_info" in folder_result and folder_result["retry_info"].get("total_attempts", 0) > 0:
                        results["retry_statistics"]["total_api_calls"] += 1
                        retry_info = folder_result["retry_info"]
                        if retry_info.get("had_retries") and retry_info.get("successful"):
                            results["retry_statistics"]["successful_retries"] += 1
                        elif retry_info.get("had_retries") and not retry_info.get("successful"):
                            results["retry_statistics"]["failed_after_retries"] += 1
                    
                    if folder_result.get("success"):
                        results["processed_folders"].append(folder_result)
                        results["total_images_generated"] += len(folder_result.get("generated_images", []))
                    else:
                        if "error" in folder_result:
                            results["errors"].append(folder_result["error"])
        
        except Exception as e:
            error_msg = f"处理过程中发生错误: {e}"
            results["errors"].append(error_msg)
            print(error_msg)
        
        return results
    
    async def save_processing_report(self, results):
        """保存处理报告"""
        # 保存JSON格式的详细结果
        json_file = self.output_dir / f"processing_results_{self.timestamp}.json"
        async with aiofiles.open(json_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(results, ensure_ascii=False, indent=2))
        
        # 保存文本格式的简要报告
        report_file = self.output_dir / f"processing_report_{self.timestamp}.txt"
        async with aiofiles.open(report_file, "w", encoding="utf-8") as f:
            report_content = (
                f"Outfit图片处理报告\n"
                f"处理时间: {datetime.fromtimestamp(self.timestamp)}\n"
                f"时间戳: {self.timestamp}\n\n"
                f"处理结果:\n"
                f"- 处理文件夹数量: {len(results['processed_folders'])}\n"
                f"- 生成图片总数: {results['total_images_generated']}\n"
                f"- 错误数量: {len(results['errors'])}\n\n"
            )
            
            # 重试统计
            retry_stats = results['retry_statistics']
            report_content += (
                f"重试统计:\n"
                f"- 总API调用次数: {retry_stats['total_api_calls']}\n"
                f"- 成功重试次数: {retry_stats['successful_retries']}\n"
                f"- 重试后仍失败次数: {retry_stats['failed_after_retries']}\n"
            )
            if retry_stats['total_api_calls'] > 0:
                success_rate = ((retry_stats['total_api_calls'] - retry_stats['failed_after_retries']) / retry_stats['total_api_calls']) * 100
                report_content += f"- 整体成功率: {success_rate:.1f}%\n"
            report_content += "\n"
            
            if results['processed_folders']:
                report_content += "成功处理的文件夹:\n"
                for folder in results['processed_folders']:
                    report_content += f"- {folder['folder_name']}: {len(folder['generated_images'])} 张图片\n"
                report_content += "\n"
            
            if results['errors']:
                report_content += "错误信息:\n"
                for error in results['errors']:
                    report_content += f"- {error}\n"
            
            await f.write(report_content)
        
        print(f"\n处理报告已保存:")
        print(f"- 详细结果: {json_file}")
        print(f"- 简要报告: {report_file}")

async def main():
    """主函数"""
    print("开始处理Outfit图片...")
    
    processor = OutfitProcessor()
    try:
        results = await processor.process_all_folders()
        await processor.save_processing_report(results)
        
        print(f"\n处理完成!")
        print(f"总共处理了 {len(results['processed_folders'])} 个文件夹")
        print(f"生成了 {results['total_images_generated']} 张图片")
        
        # 显示重试统计
        retry_stats = results['retry_statistics']
        if retry_stats['total_api_calls'] > 0:
            print(f"API调用统计: 总计{retry_stats['total_api_calls']}次, 成功重试{retry_stats['successful_retries']}次")
            if retry_stats['successful_retries'] > 0:
                print(f"重试机制帮助成功处理了 {retry_stats['successful_retries']} 个文件夹")
        
        if results['errors']:
            print(f"遇到 {len(results['errors'])} 个错误，详情请查看报告文件")
            
    finally:
        # 关闭异步客户端
        await processor.close()

if __name__ == "__main__":
    asyncio.run(main())
