import { Check, Copy } from "lucide-react";

import { Button } from "../button";
import ChatAvatar from "./chat-avatar";
import { Message } from "./chat.interface";
import Markdown from "./markdown";
import { useCopyToClipboard } from "./use-copy-to-clipboard";
import { cn } from "../lib/utils";

export default function ChatMessage(chatMessage: Message) {
  const { isCopied, copyToClipboard } = useCopyToClipboard({ timeout: 2000 });
  return (
    <div className="flex items-start gap-4 pr-5 pt-5">
      <ChatAvatar role={chatMessage.role} />
      <div className="group flex flex-1 justify-between gap-2 text-sm">
        <div
          className={cn("flex-1", {
            "text-[13px] text-gray-500 font-bold":
              chatMessage.role === "function",
          })}
        >
          <Markdown content={chatMessage.content} />
        </div>
        <Button
          onClick={() => copyToClipboard(chatMessage.content)}
          size="icon"
          variant="ghost"
          className="h-8 w-8 opacity-0 group-hover:opacity-100"
        >
          {isCopied ? (
            <Check className="h-4 w-4" />
          ) : (
            <Copy className="h-4 w-4" />
          )}
        </Button>
      </div>
    </div>
  );
}
