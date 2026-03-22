import Foundation

struct Prediction: Codable, Equatable {
    let pHome: Double
    let pDraw: Double
    let pAway: Double
    let btts: Double?
    let o25: Double?
    let egHome: Double?
    let egAway: Double?
    
    var mostLikelyOutcome: String {
        let probs = [
            ("Home", pHome),
            ("Draw", pDraw),
            ("Away", pAway)
        ]
        return probs.max(by: { $0.1 < $1.1 })?.0 ?? "Draw"
    }
    
    var mostLikelyProb: Double {
        return max(pHome, pDraw, pAway)
    }
    
    var totalProbability: Double {
        return pHome + pDraw + pAway
    }
}
