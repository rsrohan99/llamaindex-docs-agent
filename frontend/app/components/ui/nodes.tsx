import { Node } from "./chat/chat.interface";
import { useEffect, useState } from "react";

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
  selectedNodeId: string;
  onNodeClick: (nodeId: string) => void;
}

const NodeList: React.FC<NodeListProps> = ({
  nodes,
  onNodeClick,
  selectedNodeId,
}) => {
  if (nodes.length === 0) return;
  // console.log(nodes);
  // const [selectedNodeId, setSelectedNodeId] = useState(nodes[0].id || "");

  const handleNodeClick = (nodeId: string) => {
    // setSelectedNodeId(nodeId);
    onNodeClick(nodeId);
  };

  return (
    <div className="flex flex-wrap gap-2">
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
  // console.log(url);
  return (
    <div className="mt-2">
      <iframe
        src={url}
        className="w-[50vw] rounded-lg h-[80vh] border border-gray-300"
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
    const node_url = node.url.endsWith("/index")
      ? node.url.slice(0, -6)
      : node.url;
    return `https://ts.llamaindex.ai/${node_url}#${node.section}`;
  };
  const [selectedNodeUrl, setSelectedNodeUrl] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState("");

  useEffect(() => {
    // console.log("nodes updated");
    const firstNodeUrl = nodeToURL(nodes[0]);
    setSelectedNodeUrl(firstNodeUrl);
    setSelectedNodeId(nodes[0].id);
    // console.log(nodes);
  }, [nodes]);

  const handleNodeClick = (nodeId: string) => {
    const selectedNode = nodes.find((node) => node.id === nodeId);
    if (selectedNode) {
      setSelectedNodeUrl(nodeToURL(selectedNode));
      setSelectedNodeId(nodeId);
    }
  };

  return (
    <div className="p-4 pt-0 w-max">
      <NodeList
        nodes={nodes}
        onNodeClick={handleNodeClick}
        selectedNodeId={selectedNodeId}
      />
      <NodeDetails url={selectedNodeUrl} />
    </div>
  );
};
