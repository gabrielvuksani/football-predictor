import Foundation

// MARK: - Value Bets

struct ValueBetsResponse: Codable {
    let bets: [ValueBet]
}

struct ValueBet: Codable, Identifiable, Equatable {
    let matchId: Int
    let homeTeam: String
    let awayTeam: String
    let competition: String
    let date: String
    let bet: String
    let modelProb: Double
    let odds: Double
    let edge: Double
    
    var id: String { "\(matchId)-\(bet)" }
    
    var displayBet: String {
        switch bet {
        case "home": return "Home Win"
        case "draw": return "Draw"
        case "away": return "Away Win"
        default: return bet
        }
    }
    
    var edgePercentage: String {
        return String(format: "%.1f%%", edge * 100)
    }
}

// MARK: - BTTS/Over Under

struct BTTSOUResponse: Codable {
    let bttsLikely: [BTTSMatch]
    let bttsUnlikely: [BTTSMatch]
    let over25: [O25Match]
    let under25: [O25Match]
    
}

struct BTTSMatch: Codable, Identifiable, Equatable {
    let matchId: Int
    let homeTeam: String
    let awayTeam: String
    let competition: String
    let bttsProb: Double?
    
    var id: Int { matchId }
    
}

struct O25Match: Codable, Identifiable, Equatable {
    let matchId: Int
    let homeTeam: String
    let awayTeam: String
    let competition: String
    let o25Prob: Double?
    let under25Prob: Double?
    
    var id: Int { matchId }
    
}

// MARK: - Accumulators

struct AccumulatorsResponse: Codable {
    let accumulators: [Accumulator]
}

struct Accumulator: Codable, Identifiable, Equatable {
    let legs: [AccumulatorLeg]
    let odds: Double

    var id: String {
        let legIds = legs.map { $0.id }.joined(separator: "_")
        return "\(legIds)_\(odds)"
    }
}

struct AccumulatorLeg: Codable, Identifiable, Equatable {
    let matchId: Int
    let homeTeam: String
    let awayTeam: String
    let competition: String
    let outcome: String
    let probability: Double
    
    var id: String { "\(matchId)-\(outcome)" }
    
    var outcomeDisplay: String {
        switch outcome {
        case "home": return "Home Win"
        case "draw": return "Draw"
        case "away": return "Away Win"
        default: return outcome
        }
    }
}
