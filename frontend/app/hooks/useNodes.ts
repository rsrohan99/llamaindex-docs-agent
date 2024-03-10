import { useEffect, useState } from "react";
import { Node } from "../components/ui/chat/chat.interface";

export default function useNodes() {
  const [nodes, setNodes] = useState<Node[]>([]);

  useEffect(() => {
    console.log("setting nodes");
    console.log(nodes);
  }, [nodes]);

  return { nodes, setNodes };
}
