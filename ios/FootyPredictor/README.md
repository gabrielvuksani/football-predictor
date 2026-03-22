# Footy Predictor iOS App

A native SwiftUI iOS application for the Footy Predictor football prediction system. Features real-time match predictions, expert analysis, insights, and system monitoring.

## Features

### 1. Match Predictions Tab
- Browse upcoming matches with AI-generated probabilities (Home/Draw/Away)
- Filter by competition (PL, CL, EL, LA, Serie A, Bundesliga)
- Filter by timeframe (3, 7, 14, 30 days)
- Search by team name
- Pull-to-refresh functionality
- Infinite scroll pagination
- Offline cache support

### 2. Match Detail View
- Full match prediction breakdown
- Expert panel with individual predictions and confidence scores
- Head-to-head history
- Team form guide
- Expected Goals (xG) analysis
- BTTS and Over 2.5 goals probabilities
- Animated probability bars

### 3. Insights Tab
- **Value Bets**: Identifies value opportunities with edge calculations
- **BTTS/Over 2.5**: Matches with high probability BTTS and goal totals
- **Accumulators**: AI-suggested multi-leg betting combinations

### 4. System Status Tab
- Training status and accuracy metrics
- Model laboratory view (all ensemble models)
- Self-learning system status with expert weights
- System health indicators
- Real-time performance metrics

### 5. Settings Tab
- Configurable API base URL (for connecting to different backend instances)
- Network connectivity status
- Dark/Light mode support
- Cache management
- App information

## Architecture

### Models (`Models/`)
- **Match.swift**: Match data structures with predictions
- **Prediction.swift**: Probability predictions and derived values
- **Insight.swift**: Value bets, BTTS/OU, accumulator models
- **SystemStatus.swift**: Training, model lab, self-learning status

### Views (`Views/`)
- **ContentView.swift**: Main tab navigation
- **MatchListView.swift**: Scrollable match list with filters
- **MatchDetailView.swift**: Full match analysis (Experts/H2H/Form tabs)
- **MatchCardView.swift**: Reusable match card component
- **InsightsView.swift**: Value bets, BTTS/OU, accumulators
- **SystemStatusView.swift**: System health and training status
- **SettingsView.swift**: Configuration and app settings

### Services (`Services/`)
- **APIClient.swift**: URLSession-based HTTP client with:
  - Async/await support
  - Automatic retry logic (3 attempts with exponential backoff)
  - JSON decoding with snake_case conversion
  - Network error handling
  - Configurable base URL
  
- **CacheManager.swift**: UserDefaults-based caching with:
  - Codable object serialization
  - Per-endpoint cache keys
  - Clear cache functionality

### Helpers (`Helpers/`)
- **Color+Extensions.swift**: Custom app colors (light/dark mode support)
- **ProbabilityBar.swift**: Probability visualization component

## API Endpoints

The app connects to the FastAPI backend and supports all prediction endpoints:

```
GET /api/matches?days=7&page=1&limit=50
GET /api/matches/{id}
GET /api/matches/{id}/experts
GET /api/matches/{id}/h2h
GET /api/matches/{id}/form
GET /api/insights/value-bets
GET /api/insights/btts-ou
GET /api/insights/accumulators
GET /api/training/status
GET /api/model-lab
GET /api/self-learning/status
GET /api/health
```

## Setup & Running

### Prerequisites
- Xcode 15.3+
- iOS 14.0+
- Running Footy Predictor backend (FastAPI server)

### Installation Steps

1. **Clone the repository**
   ```bash
   cd /Users/gabrielvuksani/Downloads/football-predictor-main/ios/FootyPredictor
   ```

2. **Open in Xcode**
   ```bash
   open FootyPredictor.xcodeproj
   ```

3. **Configure API URL** (if not using localhost:8000)
   - Go to Settings tab
   - Update API Base URL to your backend instance
   - Default: `http://localhost:8000`

4. **Build & Run**
   - Select target device/simulator
   - Press Cmd+R or Product → Run
   - App will sign with personal team automatically

### Personal Team Signing (No Developer Account Required)

The project is configured for personal team signing:
- CODE_SIGN_STYLE = Automatic
- DEVELOPMENT_TEAM = "" (will use your default team)
- Xcode will automatically handle provisioning

### Network Configuration

For local development, ensure:
- Backend is running on `http://localhost:8000`
- Info.plist allows arbitrary loads: `NSAllowsArbitraryLoads = true`
- Device/simulator can reach the backend IP

## Technical Details

### Async/Await
All network requests use Swift's async/await with proper error handling and retry logic.

### Dark Mode Support
The app automatically adapts to system dark/light mode preferences. Custom colors scale properly:
- Background colors adjust opacity and tone
- Text colors use semantic system colors
- Accent colors defined in Assets

### Caching Strategy
- Last-fetched data cached to UserDefaults
- Cache cleared on app settings reset
- Displayed if network request fails
- Automatic population on successful API calls

### Error Handling
- Network errors caught and displayed to user
- Offline indicator in Settings
- Fallback to cached data when available
- Automatic retry with exponential backoff

## Performance Optimizations

- LazyVStack for efficient match list rendering
- Pagination support (50 matches per page)
- Pull-to-refresh for manual data refresh
- Async data loading prevents UI blocking
- JSONDecoder with optimized strategies

## File Structure

```
ios/FootyPredictor/
├── FootyPredictor.xcodeproj/
│   └── project.pbxproj
├── FootyPredictor/
│   ├── FootyPredictorApp.swift
│   ├── ContentView.swift
│   ├── Info.plist
│   ├── Models/
│   │   ├── Match.swift
│   │   ├── Prediction.swift
│   │   ├── Insight.swift
│   │   └── SystemStatus.swift
│   ├── Views/
│   │   ├── MatchListView.swift
│   │   ├── MatchDetailView.swift
│   │   ├── MatchCardView.swift
│   │   ├── InsightsView.swift
│   │   ├── SystemStatusView.swift
│   │   └── SettingsView.swift
│   ├── Services/
│   │   ├── APIClient.swift
│   │   └── CacheManager.swift
│   ├── Helpers/
│   │   ├── Color+Extensions.swift
│   │   └── ProbabilityBar.swift
│   ├── Assets.xcassets/
│   │   ├── AppIcon.appiconset/
│   │   └── AccentColor.colorset/
│   └── Preview Content/
└── README.md
```

## Key Features Implementation

### Probability Visualization
- Horizontal bar chart showing Home/Draw/Away percentages
- Color-coded (Blue/Yellow/Red)
- Animated height changes
- Team name labels

### Expert Analysis
- Sortable by confidence score
- Individual probability breakdowns
- Visual confidence indicators

### Form Guide
- Last 5 matches per team
- Result icons (Win/Draw/Loss)
- Opponent and date information
- Score tracking

### Value Betting
- Automatic edge calculation
- Model probability vs odds comparison
- Sorted by edge magnitude
- Highlighted opportunities

### Accumulator Suggestions
- Multi-leg betting combinations
- Probability multiplication
- Outcome suggestions
- Odds calculation

## Debugging

### Console Logs
- Network requests logged with timing
- Cache operations logged
- API errors displayed to user
- Network connectivity changes tracked

### Network Requests
Use Xcode's Network Link Conditioner or Charles Proxy to test:
- Slow connections
- Offline scenarios
- Failed requests

### Simulator Testing
- iOS 14.0+ simulators supported
- All features work in simulator
- Local backend access via localhost:8000

## Known Limitations

1. **No Developer Account**: Uses personal team signing (iOS 16+ may have restrictions)
2. **No Push Notifications**: All data is pull-based
3. **No Data Syncing**: Data doesn't sync across devices
4. **Limited Offline Mode**: Cached data only, no full offline functionality

## Future Enhancements

- Push notifications for match updates
- Home screen widgets
- Apple Watch companion app
- Advanced filtering (by odds, margin, etc.)
- Bet tracking and statistics
- Team-specific notifications

## Support

For issues or questions:
1. Check API connection in Settings
2. Verify backend is running
3. Review app logs in Xcode console
4. Clear cache and restart app

## License

Same as main Footy Predictor project.
