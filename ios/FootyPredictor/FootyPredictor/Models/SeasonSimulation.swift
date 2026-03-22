import Foundation

// MARK: - Season Simulation Response

struct SeasonSimulationResponse: Codable {
    let competition: String
    let teams: [SimulatedTeam]
}

struct SimulatedTeam: Codable, Identifiable {
    var id: String { team }
    let team: String
    let pTitle: Double
    let pTop4: Double
    let pRelegation: Double
    let currentPts: Int
    let currentPos: Int

}
