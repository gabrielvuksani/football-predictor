import Foundation

// MARK: - Streaks Response

struct StreaksResponse: Codable {
    let winningStreaks: [TeamStreak]?
    let losingStreaks: [TeamStreak]?
    let unbeatenRuns: [TeamStreak]?

}

struct TeamStreak: Codable, Identifiable, Equatable {
    var id: String { "\(team)-\(streak)" }
    let team: String
    let streak: Int
    let competition: String?
}
