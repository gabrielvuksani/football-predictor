import Foundation

// MARK: - Training Status
// API returns: status, active_version, expert_rankings, history, last_training, next_scheduled, message

struct TrainingStatusResponse: Codable {
    let status: String
    let activeVersion: String?
    let expertRankings: [ExpertRanking]?
    let lastTraining: String?
    let nextScheduled: String?
    let message: String?
}

struct ExpertRanking: Codable, Equatable {
    let expert: String
    let rank: Int
}

// MARK: - Model Lab
// API returns: status, active_version, models (array of strings), ensemble_weights, expert_weights, message

struct ModelLabResponse: Codable {
    let status: String?
    let activeVersion: String?
    let models: [String]
    let ensembleWeights: [EnsembleWeight]?
    let expertWeights: [EnsembleWeight]?
    let message: String?
}

struct EnsembleWeight: Codable, Equatable {
    let model: String
    let weight: Double
}

// MARK: - Self-Learning Status
// API returns: status, learning, report, message

struct SelfLearningStatusResponse: Codable {
    let status: String
    let learning: Bool
    let report: SelfLearningReport?
    let message: String?
}

struct SelfLearningReport: Codable {
    let overall: SelfLearningOverall?
    let expertRankings: [ExpertRankingEntry]?
    let driftDetected: Bool?
    let retrainRecommended: Bool?
}

struct SelfLearningOverall: Codable {
    let nPredictions: Int?
    let meanLogLoss: Double?
    let accuracy: Double?
}

struct ExpertRankingEntry: Codable {
    let expert: String?
    let rank: Int?
    let accuracy: Double?
}

// MARK: - Health
// API returns: status, app, version, timestamp

struct HealthResponse: Codable {
    let status: String
    let app: String?
    let version: String?
    let timestamp: String?
}
