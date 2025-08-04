export function StatusBadge({
  isConnected,
  serviceName,
}: {
  isConnected: boolean;
  serviceName: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground font-medium">{serviceName}:</span>
      <div className={`h-6 px-3 text-sm rounded-md flex items-center ${
        isConnected 
          ? 'bg-green-100 text-green-800 border border-green-200' 
          : 'bg-red-100 text-red-800 border border-red-200'
      }`}>
        {isConnected ? (
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full" />
            <span>Connected</span>
          </div>
        ) : (
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-red-500 rounded-full" />
            <span>Error</span>
          </div>
        )}
      </div>
    </div>
  )
}