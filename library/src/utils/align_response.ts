import OpenAI from 'openai';

/**
 * Align the response text according to the model type
 * @param model Model Name
 * @param response_msg Response Message (OpenAI.Chat.Completions.ChatCompletionMessage)
 * @returns Aligned Text or Reasoning
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function align_response_text(model: string, response_msg: any): string | undefined {
  if (model.includes('gpt-oss') || 
      model === 'openai/gpt-5-chat' || 
      model === 'anthropic/claude-sonnet-4') {
    return response_msg?.content;
  } else if (model === 'google/gemini-2.5-pro') {
    return (response_msg as OpenAI.Chat.Completions.ChatCompletionMessage & { reasoning?: string })?.reasoning || response_msg?.content;
  } else {
    return response_msg?.content;
  }
}
