import { Node } from "./chat/chat.interface";
import { useState } from "react";

interface NodeBoxProps {
  node: Node;
  onClick: () => void;
  isSelected: boolean;
}

const NodeBox: React.FC<NodeBoxProps> = ({ node, onClick, isSelected }) => {
  return (
    <div
      className={`p-3 cursor-pointer rounded-lg border border-gray-300 ${
        isSelected ? "bg-blue-200" : ""
      }`}
      onClick={onClick}
    >
      <h2 className="text-sm font-bold mb-1">{node.title}</h2>
      <p className="text-xs w-64 h-8 overflow-hidden">{node.summary}</p>
    </div>
  );
};

interface NodeListProps {
  nodes: Node[];
  onNodeClick: (nodeId: string) => void;
}

const NodeList: React.FC<NodeListProps> = ({ nodes, onNodeClick }) => {
  if (nodes.length === 0) return;
  // console.log(nodes);
  const [selectedNodeId, setSelectedNodeId] = useState(nodes[0].id || "");

  const handleNodeClick = (nodeId: string) => {
    setSelectedNodeId(nodeId);
    onNodeClick(nodeId);
  };

  return (
    <div className="flex flex-wrap gap-3">
      {nodes.map((node) => (
        <NodeBox
          key={node.id}
          node={node}
          onClick={() => handleNodeClick(node.id)}
          isSelected={selectedNodeId === node.id}
        />
      ))}
    </div>
  );
};

interface NodeDetailsProps {
  url: string;
}

const NodeDetails: React.FC<NodeDetailsProps> = ({ url }) => {
  if (!url) return;
  console.log(url);
  return (
    <div className="mt-4">
      <iframe
        src={url}
        className="w-[50vw] rounded-lg h-screen border border-gray-300"
      />
    </div>
  );
};

interface NodePreviewProps {
  nodes: Node[];
}

export const NodePreview: React.FC<NodePreviewProps> = ({ nodes }) => {
  // console.log(nodes);
  if (nodes.length === 0) return;
  const nodeToURL = (node: Node) => {
    return `https://ts.llamaindex.ai/${node.url}#${node.section}`;
  };
  const firstNodeUrl = nodeToURL(nodes[0]);
  const [selectedNodeUrl, setSelectedNodeUrl] = useState(firstNodeUrl || "");

  const handleNodeClick = (nodeId: string) => {
    const selectedNode = nodes.find((node) => node.id === nodeId);
    if (selectedNode) {
      setSelectedNodeUrl(nodeToURL(selectedNode));
    }
  };

  return (
    <div className="container mx-auto p-4 pt-0">
      <NodeList nodes={nodes} onNodeClick={handleNodeClick} />
      <NodeDetails url={selectedNodeUrl} />
    </div>
  );
};
