"use client";

import { Message, useChat } from "ai/react";
import { ChatInput, ChatMessages } from "./ui/chat";
import { useMemo } from "react";
import { transformMessages } from "./transform";
import useNodes from "../hooks/useNodes";
import { NodePreview } from "./ui/nodes";

export default function ChatSection() {
  const {
    messages,
    input,
    isLoading,
    handleSubmit,
    handleInputChange,
    reload,
    stop,
  } = useChat({
    api: process.env.NEXT_PUBLIC_CHAT_API,
  });

  const { nodes, setNodes } = useNodes();

  const mergeFunctionMessages = (messages: Message[]): Message[] => {
    // only allow the last function message to be shown
    return messages.filter(
      (msg, i) => msg.role !== "function" || i === messages.length - 1
    );
  };

  const transformedMessages = useMemo(() => {
    // return mergeFunctionMessages(transformMessages(messages));
    return transformMessages(messages, setNodes);
  }, [messages]);

  return (
    <div className="space-y-4">
      <div className="flex flex-row gap-2 w-[97vw] flex-wrap">
        <div className="w-[45vw]">
          <ChatMessages
            messages={transformedMessages}
            isLoading={isLoading}
            reload={reload}
            stop={stop}
          />
          <ChatInput
            input={input}
            handleSubmit={handleSubmit}
            handleInputChange={handleInputChange}
            isLoading={isLoading}
          />
        </div>
        {nodes.length === 0 ? (
          <></>
        ) : (
          <div className="w-[35vw]">
            <NodePreview nodes={nodes} />
          </div>
        )}
      </div>
    </div>
  );
}
