import logging
import json
from chatbox.chat_controllers.base_controller import BaseController

logger = logging.getLogger(__name__)

class VllmController(BaseController):
    default_max_tokens = 512
    chat_url = 'v1/chat/completions'
    think_start_tag = '<think>'
    think_end_tag = '</think>'

    def __init__(self, server_ip, server_port, chat_params, api_key='NA'):
        super().__init__(server_ip, server_port, chat_params)
        self.api_key = api_key
        self.in_reasoning = False

        # 新增：上下文管理器
        self.context = ContextManager(
            max_history=chat_params.get('max_history', 10),
            max_tokens=chat_params.get('context_max_tokens', 2048)
        )

    def get_headers(self):
        return {
            'User-Agent': 'python-client',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json'
        }

    def _chat(self, request_data, stream=False):
        return self.do_request('post', self.chat_url, request_data, stream=stream)

    def _deal_with_response(self, response, with_think):
        choices = response.get('choices', [])
        if not choices:
            raise Exception(
                f'No choices found in response from '
                f'{self.base_url}/{self.chat_url}: {response}'
            )
        
        message = choices[0].get('message', {})
        if not message:
            raise Exception(
                f'No message found in response from '
                f'{self.base_url}/{self.chat_url}: {response}'
            )
        
        logger.debug(f'Received response text from {self.base_url}/{self.chat_url}: {message}')
        
        # 新增：保存助手回复到上下文
        if message.get("content"):
            self.context.add_message('assistant', message['content'])
        
        if with_think:
            return {
                'reasoning_content': message.get('reasoning_content', ''),
                'content': message.get('content', '')
            }
        return {'content': message.get('content', '')}

    def _get_chat_data(self, messages, stream=False):
        chat_data = {
            "model": self.chat_params.get("model"),
            "max_tokens": int(self.chat_params.get("max_tokens", self.default_max_tokens)),
            "messages": messages,
            "stream": stream
        }
        
        if self.chat_params.get("temperature"):
            chat_data["temperature"] = float(self.chat_params.get("temperature"))
        
        logger.debug(f'Chat request data: {chat_data}')
        return chat_data

    # 修改：添加用户消息到上下文并获取完整对话历史
    def chat(self, user_input, with_think=False):
        # 添加用户消息到上下文
        self.context.add_message('user', user_input)
        # 获取完整对话历史
        messages = self.context.get_messages()
        chat_data = self._get_chat_data(messages)
        res = self._chat(chat_data)
        return self._deal_with_response(res.json(), with_think)

    def stream_chat(self, user_input):
        # 添加用户消息到上下文
        self.context.add_message('user', user_input)
        # 获取完整对话历史
        messages = self.context.get_messages()
        response = None
        
        try:
            self.in_reasoning = False
            chat_data = self._get_chat_data(messages, stream=True)
            response = self._chat(chat_data, stream=True)
            full_content = ""
            
            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue
                chunk = self._deal_with_stream_response(line)
                full_content += chunk
                yield chunk

            # 新增：将完整回复添加到上下文
            if full_content:
                self.context.add_message("assistant", full_content)
                
        except Exception as e:
            error_msg = str(e)
            raise Exception(f"Stream chat failed: {error_msg}")
        finally:
            if response:
                response.close()

    def _deal_with_stream_response(self, response):
        try:
            logger.debug(response)
            data = response.split('data: ')[1]
            if data == '[DONE]':
                return ""
            data = json.loads(data)
            delta = data.get('choices', [{}])[0].get('delta', {})
            reasoning_content = delta.get('reasoning_content')
            content = delta.get('content')
            
            if reasoning_content is not None:
                if not self.in_reasoning:
                    self.in_reasoning = True
                    return f"{self.think_start_tag}{reasoning_content}"
                else:
                    return reasoning_content
            elif content is not None:
                if self.in_reasoning:
                    self.in_reasoning = False
                    return f"{self.think_end_tag}{content}"
                else:
                    return content
            else:
                return ""
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Deal with stream response failed: {error_msg}")
            return ""

    # 新增：上下文管理方法
    def reset_context(self):
        """清空对话上下文"""
        self.context.reset()
    
    def get_context(self):
        """获取当前对话上下文"""
        return self.context.get_messages()
    
    def set_system_message(self, message):
        """设置系统消息"""
        self.context.set_system_message(message)


class ContextManager:
    def __init__(self, max_history=10, max_tokens=2048):
        self.messages = []
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.system_message = None

    def add_message(self, role, content):
        if role == "system":
            self.set_system_message(content)
        else:
            self.messages.append({'role': role, 'content': content})
            self._prune_messages()

    def set_system_message(self, message):
        """设置系统消息"""
        self.system_message = message

    def get_messages(self):
        """获取完整对话消息列表"""
        messages = []

        # 添加系统消息（如果有）
        if self.system_message:
            messages.append({'role': 'system', 'content': self.system_message})

        # 添加对话历史
        messages.extend(self.messages)
        return messages

    def reset(self):
        """重置对话上下文"""
        self.messages = []

    def _prune_messages(self):
        """修剪消息历史，确保不超过限制"""
        # 1. 数量限制
        if len(self.messages) > self.max_history:
            # 保留最近的 max_history 条消息
            self.messages = self.messages[-self.max_history:]

        # 2. Token 限制（简化实现）
        # 实际应用中应使用 tokenizer 计算 token 数
        total_length = sum(len(msg['content']) for msg in self.messages)
        if total_length > self.max_tokens:
            # 移除最早的消息直到满足 token 限制
            while total_length > self.max_tokens and len(self.messages) > 1:
                removed = self.messages.pop(0)
                total_length -= len(removed["content"])
