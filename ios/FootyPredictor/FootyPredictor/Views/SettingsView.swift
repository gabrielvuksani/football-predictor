import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var apiClient: APIClient
    @State private var isEditing = false
    @State private var tempURL = ""
    @State private var showClearCacheConfirmation = false
    @AppStorage("api_base_url") private var savedBaseURL: String = APIClient.defaultBaseURL
    @Environment(\.colorScheme) var colorScheme

    private var appVersion: String {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? ""
        return build.isEmpty ? version : "\(version) (\(build))"
    }
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color.appBackground.ignoresSafeArea()
                
                Form {
                    Section(header: Text("API Configuration")) {
                        VStack(alignment: .leading, spacing: 8) {
                            Label("API Server URL", systemImage: "server.rack")
                                .font(.caption)
                                .foregroundColor(.gray)

                            HStack {
                                TextField("http://localhost:8000", text: $savedBaseURL)
                                    .textFieldStyle(.roundedBorder)
                                    .font(.caption)
                                    .textInputAutocapitalization(.never)
                                    .disableAutocorrection(true)
                                    .keyboardType(.URL)
                                    .onChange(of: savedBaseURL) { newValue in
                                        apiClient.baseURL = newValue
                                    }

                                Button(action: resetURL) {
                                    Image(systemName: "arrow.counterclockwise.circle")
                                        .foregroundColor(.gray)
                                }
                            }

                            Text("Current: \(savedBaseURL)")
                                .font(.caption2)
                                .foregroundColor(.gray)
                        }
                    }
                    
                    Section(header: Text("Network Status")) {
                        HStack {
                            HStack(spacing: 8) {
                                Circle()
                                    .fill(apiClient.isOnline ? Color.successGreen : Color.dangerRed)
                                    .frame(width: 10, height: 10)
                                Text(apiClient.isOnline ? "Online" : "Offline")
                                    .font(.caption)
                            }
                            Spacer()
                            Text(apiClient.isOnline ? "Connected" : "No Connection")
                                .font(.caption2)
                                .foregroundColor(.gray)
                        }
                    }
                    
                    Section(header: Text("Appearance")) {
                        HStack {
                            Image(systemName: "moon.stars.fill")
                                .foregroundColor(.accentBlue)
                            Text("Dark Mode")
                            Spacer()
                            Text(colorScheme == .dark ? "On" : "Off")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    }
                    
                    Section(header: Text("Cache")) {
                        Button(action: { showClearCacheConfirmation = true }) {
                            HStack {
                                Image(systemName: "trash.fill")
                                    .foregroundColor(.dangerRed)
                                Text("Clear Cache")
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .foregroundColor(.gray)
                            }
                            .foregroundColor(.primary)
                        }
                        .confirmationDialog(
                            "Clear Cache",
                            isPresented: $showClearCacheConfirmation,
                            titleVisibility: .visible
                        ) {
                            Button("Clear All Cached Data", role: .destructive) {
                                clearCache()
                            }
                            Button("Cancel", role: .cancel) {}
                        } message: {
                            Text("This will remove all cached match data, predictions, and insights. You will need to reload data from the server.")
                        }
                    }
                    
                    Section(header: Text("System")) {
                        NavigationLink(destination: SystemStatusView()) {
                            HStack {
                                Image(systemName: "server.rack")
                                    .foregroundColor(.accentBlue)
                                Text("System Status")
                                Spacer()
                            }
                            .foregroundColor(.primary)
                        }
                    }

                    Section(header: Text("About")) {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Version")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                                Spacer()
                                Text(appVersion)
                                    .font(.caption2)
                                    .foregroundColor(.gray)
                            }
                            
                            HStack {
                                Text("App Name")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                                Spacer()
                                Text("Footy Predictor")
                                    .font(.caption2)
                                    .foregroundColor(.gray)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Settings")
            .onAppear {
                savedBaseURL = UserDefaults.standard.string(forKey: APIClient.baseURLKey) ?? APIClient.defaultBaseURL
                apiClient.baseURL = savedBaseURL
            }
        }
    }
    
    private func resetURL() {
        savedBaseURL = APIClient.defaultBaseURL
        apiClient.baseURL = APIClient.defaultBaseURL
    }
    
    private func clearCache() {
        CacheManager.shared.clearAllCache()
    }
}

#Preview {
    SettingsView()
        .environmentObject(APIClient())
}
