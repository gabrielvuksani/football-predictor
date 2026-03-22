import Foundation

class CacheManager {
    static let shared = CacheManager()

    private let defaults = UserDefaults.standard
    private let fileManager = FileManager.default
    private let keyPrefix = "cache_"
    private let timestampSuffix = "_ts"

    /// Default TTL: 30 minutes
    var defaultTTL: TimeInterval = 30 * 60

    private lazy var cacheDirectory: URL? = {
        return fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first?
            .appendingPathComponent("FootyPredictorCache")
    }()

    private init() {
        setupCacheDirectory()
    }

    private func setupCacheDirectory() {
        guard let cacheDir = cacheDirectory else { return }
        try? fileManager.createDirectory(at: cacheDir, withIntermediateDirectories: true)
    }

    private func prefixedKey(_ key: String) -> String {
        return keyPrefix + key
    }

    private func timestampKey(_ key: String) -> String {
        return keyPrefix + key + timestampSuffix
    }

    // MARK: - Codable Caching

    func cache<T: Codable>(_ object: T, forKey key: String) {
        let pKey = prefixedKey(key)
        do {
            let data = try JSONEncoder().encode(object)
            defaults.set(data, forKey: pKey)
            defaults.set(Date().timeIntervalSince1970, forKey: timestampKey(key))
        } catch {
            print("Cache encoding error: \(error)")
        }
    }

    func retrieveCache<T: Codable>(forKey key: String, type: T.Type, ttl: TimeInterval? = nil) -> T? {
        let pKey = prefixedKey(key)
        let tsKey = timestampKey(key)
        let effectiveTTL = ttl ?? defaultTTL

        // Check TTL
        let cachedTimestamp = defaults.double(forKey: tsKey)
        if cachedTimestamp > 0 {
            let elapsed = Date().timeIntervalSince1970 - cachedTimestamp
            if elapsed > effectiveTTL {
                // Expired - remove and return nil
                defaults.removeObject(forKey: pKey)
                defaults.removeObject(forKey: tsKey)
                return nil
            }
        }

        guard let data = defaults.data(forKey: pKey) else { return nil }
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            print("Cache decoding error: \(error)")
            return nil
        }
    }

    func clearCache(forKey key: String) {
        defaults.removeObject(forKey: prefixedKey(key))
        defaults.removeObject(forKey: timestampKey(key))
    }

    func clearAllCache() {
        let allKeys = defaults.dictionaryRepresentation().keys
        for key in allKeys where key.hasPrefix(keyPrefix) {
            defaults.removeObject(forKey: key)
        }
    }

    // MARK: - Cache Keys

    static let matchListCacheKey = "matchList"
    static let matchDetailCacheKeyPrefix = "matchDetail_"
    static let matchExpertsCacheKeyPrefix = "matchExperts_"
    static let h2hCacheKeyPrefix = "h2h_"
    static let formCacheKeyPrefix = "form_"
    static let valueBetsCacheKey = "valueBets"
    static let bttsouCacheKey = "bttsou"
    static let accumulatorsCacheKey = "accumulators"
    static let trainingStatusCacheKey = "trainingStatus"
    static let modelLabCacheKey = "modelLab"
    static let selfLearningCacheKey = "selfLearning"
    static let healthCacheKey = "health"
}
