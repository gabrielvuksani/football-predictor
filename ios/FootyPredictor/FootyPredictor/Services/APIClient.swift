import Foundation

enum APIError: LocalizedError {
    case invalidURL
    case networkError(Error)
    case decodingError(DecodingError)
    case invalidResponse(Int)
    case serverError(String)
    case offline
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .decodingError(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .invalidResponse(let code):
            return "Invalid response status: \(code)"
        case .serverError(let msg):
            return "Server error: \(msg)"
        case .offline:
            return "Network connection unavailable"
        }
    }
}

class APIClient: ObservableObject {
    // Default to GitHub Pages static API for zero-config experience.
    // Users can point to their own server in Settings.
    static let defaultBaseURL = "http://localhost:8000"
    static let baseURLKey = "api_base_url"

    @Published var baseURL: String {
        didSet {
            UserDefaults.standard.set(baseURL, forKey: APIClient.baseURLKey)
        }
    }
    @Published var isOnline: Bool = true

    private let session: URLSession
    private let decoder: JSONDecoder
    private let maxRetries = 3

    init() {
        self.baseURL = UserDefaults.standard.string(forKey: APIClient.baseURLKey) ?? APIClient.defaultBaseURL

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 15
        config.waitsForConnectivity = false
        self.session = URLSession(configuration: config)

        self.decoder = JSONDecoder()
        self.decoder.keyDecodingStrategy = .convertFromSnakeCase
    }
    
    // MARK: - Matches
    
    func fetchMatches(days: Int = 14, competition: String? = nil, status: String = "UPCOMING", page: Int = 1, limit: Int = 50) async throws -> MatchListResponse {
        var queryItems = [
            URLQueryItem(name: "days", value: String(days)),
            URLQueryItem(name: "page", value: String(page)),
            URLQueryItem(name: "limit", value: String(limit))
        ]
        if let competition = competition, !competition.isEmpty {
            queryItems.append(URLQueryItem(name: "competition", value: competition))
        }
        let url = buildURL(path: "/api/matches", queryItems: queryItems)
        return try await fetchWithRetry(url, response: MatchListResponse.self)
    }
    
    func fetchMatchDetail(matchId: Int, model: String = "v12_analyst") async throws -> MatchDetail {
        let path = "/api/matches/\(matchId)"
        let url = buildURL(path: path, queryItems: [
            URLQueryItem(name: "model", value: model)
        ])
        return try await fetchWithRetry(url, response: MatchDetail.self)
    }
    
    func fetchMatchExperts(matchId: Int) async throws -> ExpertPredictionsResponse {
        let path = "/api/matches/\(matchId)/experts"
        let url = buildURL(path: path)
        return try await fetchWithRetry(url, response: ExpertPredictionsResponse.self)
    }
    
    func fetchH2H(matchId: Int) async throws -> H2HResponse {
        let path = "/api/matches/\(matchId)/h2h"
        let url = buildURL(path: path)
        return try await fetchWithRetry(url, response: H2HResponse.self)
    }
    
    func fetchForm(matchId: Int) async throws -> FormResponse {
        let path = "/api/matches/\(matchId)/form"
        let url = buildURL(path: path)
        return try await fetchWithRetry(url, response: FormResponse.self)
    }
    
    // MARK: - Insights
    
    func fetchValueBets(minEdge: Double = 0.03) async throws -> ValueBetsResponse {
        let path = "/api/insights/value-bets"
        let url = buildURL(path: path, queryItems: [
            URLQueryItem(name: "min_edge", value: String(minEdge))
        ])
        return try await fetchWithRetry(url, response: ValueBetsResponse.self)
    }
    
    func fetchBTTSOU() async throws -> BTTSOUResponse {
        let path = "/api/insights/btts-ou"
        let url = buildURL(path: path)
        return try await fetchWithRetry(url, response: BTTSOUResponse.self)
    }
    
    func fetchAccumulators() async throws -> AccumulatorsResponse {
        let path = "/api/insights/accumulators"
        let url = buildURL(path: path)
        return try await fetchWithRetry(url, response: AccumulatorsResponse.self)
    }
    
    // MARK: - Season Simulation

    func fetchSeasonSimulation(comp: String) async throws -> SeasonSimulationResponse {
        let path = "/api/season-simulation/\(comp)"
        let url = buildURL(path: path)
        return try await fetchWithRetry(url, response: SeasonSimulationResponse.self)
    }

    // MARK: - Team Profile

    func fetchTeamProfile(teamName: String) async throws -> TeamProfileResponse {
        let encoded = teamName.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? teamName
        let path = "/api/team/\(encoded)/profile"
        let url = buildURL(path: path)
        return try await fetchWithRetry(url, response: TeamProfileResponse.self)
    }

    // MARK: - Streaks

    func fetchStreaks() async throws -> StreaksResponse {
        let path = "/api/streaks"
        let url = buildURL(path: path)
        return try await fetchWithRetry(url, response: StreaksResponse.self)
    }

    // MARK: - System Status
    
    func fetchTrainingStatus() async throws -> TrainingStatusResponse {
        let path = "/api/training/status"
        let url = buildURL(path: path)
        return try await fetchWithRetry(url, response: TrainingStatusResponse.self)
    }
    
    func fetchModelLab() async throws -> ModelLabResponse {
        let path = "/api/model-lab"
        let url = buildURL(path: path)
        return try await fetchWithRetry(url, response: ModelLabResponse.self)
    }
    
    func fetchSelfLearningStatus() async throws -> SelfLearningStatusResponse {
        let path = "/api/self-learning/status"
        let url = buildURL(path: path)
        return try await fetchWithRetry(url, response: SelfLearningStatusResponse.self)
    }
    
    func fetchHealth() async throws -> HealthResponse {
        let path = "/api/health"
        let url = buildURL(path: path)
        return try await fetchWithRetry(url, response: HealthResponse.self)
    }
    
    // MARK: - Private Helpers
    
    private func buildURL(path: String, queryItems: [URLQueryItem] = []) -> URL {
        guard let baseURL = URL(string: baseURL) else { return URL(fileURLWithPath: "") }
        var components = URLComponents(url: baseURL.appendingPathComponent(""), resolvingAgainstBaseURL: false)!
        components.path = path
        components.queryItems = queryItems.isEmpty ? nil : queryItems
        return components.url ?? URL(fileURLWithPath: "")
    }
    
    private func fetchWithRetry<T: Decodable>(_ url: URL, response: T.Type) async throws -> T {
        var lastError: Error?
        
        for attempt in 0..<maxRetries {
            do {
                let (data, urlResponse) = try await session.data(from: url)
                
                guard let httpResponse = urlResponse as? HTTPURLResponse else {
                    throw APIError.invalidResponse(0)
                }
                
                guard (200...299).contains(httpResponse.statusCode) else {
                    if httpResponse.statusCode >= 500 {
                        lastError = APIError.invalidResponse(httpResponse.statusCode)
                        if attempt < maxRetries - 1 {
                            try await Task.sleep(nanoseconds: UInt64(pow(2.0, Double(attempt)) * 1_000_000_000))
                            continue
                        }
                    }
                    throw APIError.invalidResponse(httpResponse.statusCode)
                }
                
                do {
                    return try decoder.decode(T.self, from: data)
                } catch let decodingError as DecodingError {
                    throw APIError.decodingError(decodingError)
                }
            } catch let error as URLError {
                if error.code == .notConnectedToInternet || error.code == .networkConnectionLost {
                    DispatchQueue.main.async { self.isOnline = false }
                    lastError = APIError.offline
                } else {
                    lastError = APIError.networkError(error)
                }
                
                if attempt < maxRetries - 1 {
                    try await Task.sleep(nanoseconds: UInt64(pow(2.0, Double(attempt)) * 1_000_000_000))
                } else {
                    throw lastError ?? APIError.offline
                }
            } catch {
                throw error
            }
        }
        
        throw lastError ?? APIError.offline
    }
}
