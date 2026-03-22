import Foundation

// MARK: - Match List Response

struct MatchListResponse: Codable {
    let matches: [Match]
    let page: Int
    let limit: Int
    let totalCount: Int
    let hasMore: Bool
}

// MARK: - Match

struct Match: Codable, Identifiable, Equatable {
    let matchId: Int
    let homeTeam: String
    let awayTeam: String
    let competition: String
    let utcDate: String
    let pHome: Double?
    let pDraw: Double?
    let pAway: Double?
    let btts: Double?
    let o25: Double?
    let egHome: Double?
    let egAway: Double?
    // v12 new fields
    let upsetRisk: Double?
    let confidence: Double?
    let modelAgreement: Double?
    let xgHome: Double?
    let xgAway: Double?
    let valueEdge: Double?

    var id: Int { matchId }

    /// Upset risk level classification
    var upsetLevel: String {
        guard let risk = upsetRisk else { return "unknown" }
        if risk > 0.6 { return "high" }
        if risk > 0.4 { return "medium" }
        return "low"
    }

    /// Whether model found value vs market odds
    var hasValue: Bool {
        guard let edge = valueEdge else { return false }
        return abs(edge) > 0.03
    }
    
    var dateFormatted: String {
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: utcDate + "Z") {
            let displayFormatter = DateFormatter()
            displayFormatter.dateFormat = "MMM d, HH:mm"
            return displayFormatter.string(from: date)
        }
        return utcDate
    }
    
    var timeUntilMatch: String {
        let formatter = ISO8601DateFormatter()
        guard let date = formatter.date(from: utcDate + "Z") else { return "N/A" }
        let now = Date()
        let interval = date.timeIntervalSince(now)
        
        if interval < 0 { return "Live" }
        let hours = Int(interval / 3600)
        let minutes = Int((interval.truncatingRemainder(dividingBy: 3600)) / 60)
        
        if hours > 0 {
            return "\(hours)h \(minutes)m"
        } else {
            return "\(minutes)m"
        }
    }
}

// MARK: - Match Detail

struct MatchDetail: Codable, Identifiable {
    let matchId: Int
    let homeTeam: String
    let awayTeam: String
    let competition: String
    let utcDate: String
    let status: String
    let score: MatchScore?
    let prediction: Prediction?
    let odds: OddsData?
    let elo: EloData?
    let poisson: PoissonData?
    
    var id: Int { matchId }
    
}

struct MatchScore: Codable, Equatable {
    let home: Int
    let away: Int
}

struct OddsData: Codable, Equatable {
    let home: Double
    let draw: Double
    let away: Double
}

struct EloData: Codable, Equatable {
    let home: Double?
    let away: Double?
}

struct PoissonData: Codable, Equatable {
    let home: Double
    let away: Double
}

// MARK: - Expert Predictions

struct ExpertPredictionsResponse: Codable {
    let experts: [String: ExpertPrediction]
}

struct ExpertPrediction: Codable, Identifiable, Equatable {
    let probs: ProbabilityData
    let confidence: Double

    // Stable identity: derived from content hash rather than random UUID.
    // Using UUID().uuidString as default means each JSON decode creates a new
    // identity, causing SwiftUI ForEach to treat every item as new on refresh
    // and triggering full view re-creation. Content-based ID is stable across decodes.
    var id: String {
        "\(probs.home)-\(probs.draw)-\(probs.away)-\(confidence)"
    }

    // Custom Equatable that ignores the computed id (compare content only)
    static func == (lhs: ExpertPrediction, rhs: ExpertPrediction) -> Bool {
        lhs.probs == rhs.probs && lhs.confidence == rhs.confidence
    }

    // Custom CodingKeys to exclude id from decoding
    enum CodingKeys: String, CodingKey {
        case probs, confidence
    }
}

struct ProbabilityData: Codable, Equatable {
    let home: Double
    let draw: Double
    let away: Double
}

// MARK: - H2H History

struct H2HResponse: Codable {
    let h2h: [H2HMatch]?
    let summary: H2HSummary?
}

struct H2HSummary: Codable {
    let homeWins: Int?
    let awayWins: Int?
    let draws: Int?
}

struct H2HMatch: Codable, Equatable {
    let date: String?
    let homeTeam: String
    let awayTeam: String
    let homeGoals: Int?
    let awayGoals: Int?

    var scoreFormatted: String {
        let hg = homeGoals ?? 0
        let ag = awayGoals ?? 0
        return "\(hg) - \(ag)"
    }
}

// MARK: - Form Guide

struct FormResponse: Codable {
    let homeForm: [FormMatch]
    let awayForm: [FormMatch]
}

struct FormMatch: Codable, Equatable {
    let date: String?
    let opponent: String
    let homeGoals: Int?
    let awayGoals: Int?
    let result: String
    
    var resultIcon: String {
        switch result {
        case "W": return "checkmark.circle.fill"
        case "D": return "minus.circle.fill"
        case "L": return "xmark.circle.fill"
        default: return "circle"
        }
    }
    
    var resultColor: String {
        switch result {
        case "W": return "green"
        case "D": return "yellow"
        case "L": return "red"
        default: return "gray"
        }
    }
}
