import { useState } from "react";
import { Node } from "../components/ui/chat/chat.interface";

export default function useNodes() {
  const [nodes, setNodes] = useState<Node[]>([]);

  return { nodes, setNodes };
}
