import { useState, useMemo, useEffect } from 'react'
import type { TreeNode } from '@/types'
import { cn } from '@/lib/utils'
import {
  Folder,
  FolderOpen,
  ChevronRight,
  ChevronDown,
  Search,
  X,
  ChevronsDownUp,
  AlertTriangle,
} from 'lucide-react'

interface DirectoryTreeProps {
  nodes: TreeNode[]
  selectedPath: string | null
  onSelect: (path: string, type: string) => void
}

// Font Awesome Pro 7.1.0 square-letter icon paths (commercial license).
const SQUARE_LETTER_PATHS: Record<string, string> = {
  table: 'M64 80c-8.8 0-16 7.2-16 16l0 320c0 8.8 7.2 16 16 16l320 0c8.8 0 16-7.2 16-16l0-320c0-8.8-7.2-16-16-16L64 80zM0 96C0 60.7 28.7 32 64 32l320 0c35.3 0 64 28.7 64 64l0 320c0 35.3-28.7 64-64 64L64 480c-35.3 0-64-28.7-64-64L0 96zm136 48l176 0c13.3 0 24 10.7 24 24s-10.7 24-24 24l-64 0 0 152c0 13.3-10.7 24-24 24s-24-10.7-24-24l0-152-64 0c-13.3 0-24-10.7-24-24s10.7-24 24-24z',
  view: 'M64 80c-8.8 0-16 7.2-16 16l0 320c0 8.8 7.2 16 16 16l320 0c8.8 0 16-7.2 16-16l0-320c0-8.8-7.2-16-16-16L64 80zM0 96C0 60.7 28.7 32 64 32l320 0c35.3 0 64 28.7 64 64l0 320c0 35.3-28.7 64-64 64L64 480c-35.3 0-64-28.7-64-64L0 96zm157.5 61.3L224 290.3 290.5 157.3c5.9-11.9 20.3-16.7 32.2-10.7s16.7 20.3 10.7 32.2l-88 176c-4.1 8.1-12.4 13.3-21.5 13.3s-17.4-5.1-21.5-13.3l-88-176c-5.9-11.9-1.1-26.3 10.7-32.2s26.3-1.1 32.2 10.7z',
  snapshot: 'M64 80c-8.8 0-16 7.2-16 16l0 320c0 8.8 7.2 16 16 16l320 0c8.8 0 16-7.2 16-16l0-320c0-8.8-7.2-16-16-16L64 80zM0 96C0 60.7 28.7 32 64 32l320 0c35.3 0 64 28.7 64 64l0 320c0 35.3-28.7 64-64 64L64 480c-35.3 0-64-28.7-64-64L0 96zm202.5 48l69.5 0c13.3 0 24 10.7 24 24s-10.7 24-24 24l-69.5 0c-10.2 0-18.5 8.3-18.5 18.5 0 9.3 6.9 17.2 16.2 18.3l53.6 6.7c33.3 4.2 58.2 32.4 58.2 66 0 36.7-29.8 66.5-66.5 66.5L168 368c-13.3 0-24-10.7-24-24s10.7-24 24-24l77.5 0c10.2 0 18.5-8.3 18.5-18.5 0-9.3-6.9-17.2-16.2-18.3l-53.6-6.7c-33.3-4.2-58.2-32.4-58.2-66 0-36.7 29.8-66.5 66.5-66.5z',
  replica: 'M64 80c-8.8 0-16 7.2-16 16l0 320c0 8.8 7.2 16 16 16l320 0c8.8 0 16-7.2 16-16l0-320c0-8.8-7.2-16-16-16L64 80zM0 96C0 60.7 28.7 32 64 32l320 0c35.3 0 64 28.7 64 64l0 320c0 35.3-28.7 64-64 64L64 480c-35.3 0-64-28.7-64-64L0 96zm168 48l80 0c39.8 0 72 32.2 72 72 0 28.9-17 53.8-41.6 65.3l30.2 50.3c6.8 11.4 3.1 26.1-8.2 32.9s-26.1 3.1-32.9-8.2l-41-68.3-34.4 0 0 56c0 13.3-10.7 24-24 24s-24-10.7-24-24l0-176c0-13.3 10.7-24 24-24zm72 96l8 0c13.3 0 24-10.7 24-24s-10.7-24-24-24l-56 0 0 48 48 0z',
}

function KindBadge({ kind }: { kind: string }) {
  const path = SQUARE_LETTER_PATHS[kind]
  if (!path) return null
  return (
    <svg viewBox="0 0 448 512" className="h-3.5 w-3.5 shrink-0 text-muted-foreground/80" aria-hidden="true">
      <path fill="currentColor" d={path} />
    </svg>
  )
}

function getDirectoryIcon(isOpen: boolean) {
  return isOpen
    ? <FolderOpen className="h-3.5 w-3.5 text-k-yellow shrink-0" />
    : <Folder className="h-3.5 w-3.5 text-k-yellow shrink-0" />
}

function countDescendants(node: TreeNode): number {
  if (node.kind !== 'directory' || node.entries.length === 0) return 0
  return node.entries.reduce((sum, child) => sum + 1 + countDescendants(child), 0)
}

function countAllNodes(nodes: TreeNode[]): number {
  return nodes.reduce(
    (sum, n) => sum + 1 + (n.kind === 'directory' ? countAllNodes(n.entries) : 0),
    0,
  )
}

function nodeMatchesFilter(node: TreeNode, q: string): boolean {
  if (node.name.toLowerCase().includes(q)) return true
  if (node.kind === 'directory') return node.entries.some(c => nodeMatchesFilter(c, q))
  return false
}

function TreeItem({ node, level, selectedPath, onSelect, filter, collapsedAll }: {
  node: TreeNode; level: number; selectedPath: string | null
  onSelect: (path: string, type: string) => void; filter: string; collapsedAll: number
}) {
  const [manualOpen, setManualOpen] = useState<boolean | null>(null)
  const isDirectory = node.kind === 'directory'
  const hasChildren = isDirectory && node.entries.length > 0
  const descendantCount = useMemo(() => countDescendants(node), [node])
  const hasErrors = !isDirectory && node.error_count > 0

  useEffect(() => {
    if (collapsedAll > 0) setManualOpen(false)
  }, [collapsedAll])

  const isOpen = filter
    ? true
    : manualOpen !== null
      ? manualOpen
      : level === 0

  const isSelected = selectedPath === node.path
  if (filter && !nodeMatchesFilter(node, filter)) return null

  const handleClick = () => {
    if (isDirectory && hasChildren) setManualOpen(!isOpen)
    onSelect(node.path, node.kind)
  }

  return (
    <div>
      <button
        className={cn(
          'group flex items-center gap-1.5 w-full rounded-md py-1 px-2 text-left transition-colors',
          isSelected
            ? 'bg-primary/10 text-foreground'
            : 'text-muted-foreground hover:bg-accent hover:text-foreground',
        )}
        style={{ paddingLeft: `${level * 12 + 8}px` }}
        onClick={handleClick}
        title={`${node.kind}: ${node.path}`}
      >
        {isDirectory && hasChildren ? (
          <span className="w-3.5 h-3.5 flex items-center justify-center shrink-0">
            {isOpen
              ? <ChevronDown className="h-3 w-3 text-muted-foreground" />
              : <ChevronRight className="h-3 w-3 text-muted-foreground" />}
          </span>
        ) : (
          <span className="w-3.5 h-3.5 shrink-0" />
        )}

        {isDirectory
          ? getDirectoryIcon(isOpen)
          : <span className="w-3.5 h-3.5 shrink-0" />}
        <span className="flex-1 text-[13px] truncate">{node.name}</span>

        {hasErrors && !isDirectory && (
          <span className="flex items-center gap-0.5 text-[10px] text-destructive shrink-0" title={`${node.error_count} errors`}>
            <AlertTriangle className="h-2.5 w-2.5" />
          </span>
        )}

        {isDirectory && descendantCount > 0 && (
          <span className="text-[10px] text-muted-foreground/50 tabular-nums shrink-0">
            {descendantCount}
          </span>
        )}

        {!isDirectory && <KindBadge kind={node.kind} />}

      </button>

      {isDirectory && hasChildren && isOpen && (
        <div>
          {node.entries.map((child) => (
            <TreeItem
              key={child.path}
              node={child}
              level={level + 1}
              selectedPath={selectedPath}
              onSelect={onSelect}
              filter={filter}
              collapsedAll={collapsedAll}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export function DirectoryTree({ nodes, selectedPath, onSelect }: DirectoryTreeProps) {
  const [filter, setFilter] = useState('')
  const [collapsedAll, setCollapsedAll] = useState(0)
  const totalCount = useMemo(() => countAllNodes(nodes), [nodes])
  const showFilter = totalCount >= 10
  const q = filter.toLowerCase()

  if (nodes.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <Folder className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p className="text-xs">No directories or tables found</p>
        <p className="text-[11px] mt-1 text-muted-foreground">
          Create tables using the Python SDK
        </p>
      </div>
    )
  }

  return (
    <div>
      {showFilter && (
        <div className="px-2 pb-1.5 flex items-center gap-1">
          <div className="relative flex-1">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground/50" />
            <input
              type="text"
              value={filter}
              onChange={e => setFilter(e.target.value)}
              placeholder="Filter…"
              className="h-6 w-full pl-6 pr-6 text-[11px] rounded border border-border/40 bg-background/50 text-foreground placeholder:text-muted-foreground/40 focus:outline-none focus:ring-1 focus:ring-ring/30"
            />
            {filter && (
              <button onClick={() => setFilter('')} className="absolute right-1.5 top-1/2 -translate-y-1/2">
                <X className="h-3 w-3 text-muted-foreground/50 hover:text-foreground" />
              </button>
            )}
          </div>
          <button
            onClick={() => setCollapsedAll(c => c + 1)}
            className="h-6 w-6 flex items-center justify-center rounded border border-border/40 bg-background/50 text-muted-foreground/50 hover:text-foreground transition-colors shrink-0"
            title="Collapse all"
          >
            <ChevronsDownUp className="h-3 w-3" />
          </button>
        </div>
      )}
      <div className="space-y-px">
        {nodes.map((node) => (
          <TreeItem
            key={node.path}
            node={node}
            level={0}
            selectedPath={selectedPath}
            onSelect={onSelect}
            filter={q}
            collapsedAll={collapsedAll}
          />
        ))}
      </div>
    </div>
  )
}
