import { Button } from '@/components/ui/button';

interface RatingBarProps {
  handleRating: (rating: 'good' | 'bad') => void;
  showRating: boolean;
}

export function RatingBar({ handleRating, showRating }: RatingBarProps) {
  return (
    <div 
      className={`max-w-6xl mx-auto px-4 mt-6 overflow-hidden transition-all duration-500 ease-in-out ${
        showRating ? 'max-h-40sm:max-h-20 opacity-100' : 'max-h-0 opacity-0'
      }`}
    >
      <div className="bg-gray-50 border border-gray-200 rounded-lg px-6 py-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <span className="text-gray-800 font-medium text-center sm:text-left">Do you like the results?</span>
          <div className="flex gap-3 w-full sm:w-auto sm:flex-shrink-0">
            <Button
              onClick={() => handleRating('good')}
              variant="outline"
              size="sm"
              className="flex-1 sm:flex-none w-full sm:w-auto sm:w-30 border-gray-300 hover:bg-gray-100"
            >
              Yes
            </Button>
            <Button
              onClick={() => handleRating('bad')}
              variant="outline"
              size="sm"
              className="flex-1 sm:flex-none w-full sm:w-auto sm:w-30 border-gray-300 hover:bg-gray-100"
            >
              No
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}