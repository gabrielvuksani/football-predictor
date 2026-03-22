import Foundation

// MARK: - Team Profile Response

struct TeamProfileResponse: Codable {
    let team: String
    let elo: Double?
    let form: [String]?
    let goalsScored: Int?
    let goalsConceded: Int?
    let recentMatches: [TeamMatch]?
    let upcomingFixtures: [TeamFixture]?

}

struct TeamMatch: Codable, Identifiable, Equatable {
    var id: String { "\(date)-\(opponent)" }
    let date: String
    let opponent: String
    let result: String
    let goalsFor: Int?
    let goalsAgainst: Int?
    let competition: String?

}

struct TeamFixture: Codable, Identifiable, Equatable {
    var id: String { "\(date)-\(opponent)" }
    let date: String
    let opponent: String
    let venue: String?
    let competition: String?
}
