import SwiftUI

extension Color {
    // MARK: - Backgrounds (match web design system)
    static let appBackground = Color(UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 0.02, green: 0.02, blue: 0.03, alpha: 1.0) // #050507
            : UIColor(red: 0.97, green: 0.97, blue: 0.98, alpha: 1.0) // #f8f9fa
    })

    static let cardBackground = Color(UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.04) // surface-1
            : UIColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
    })

    static let surfaceRaised = Color(UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 0.065) // surface-2
            : UIColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.08)
    })

    // MARK: - Accent colors (unified with web)
    static let accentIndigo = Color(red: 99/255, green: 102/255, blue: 241/255)  // #6366f1
    static let accentBlue = Color(red: 59/255, green: 130/255, blue: 246/255)    // #3b82f6

    // MARK: - Semantic colors (match web)
    static let successGreen = Color(red: 16/255, green: 185/255, blue: 129/255)  // #10b981
    static let warningYellow = Color(red: 245/255, green: 158/255, blue: 11/255) // #f59e0b
    static let dangerRed = Color(red: 239/255, green: 68/255, blue: 68/255)      // #ef4444
    static let infoCyan = Color(red: 6/255, green: 182/255, blue: 212/255)       // #06b6d4
    static let neutralGray = Color(red: 0.6, green: 0.6, blue: 0.6)

    // MARK: - Text colors
    static let textPrimary = Color(UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 240/255, green: 240/255, blue: 245/255, alpha: 1.0) // #f0f0f5
            : UIColor(red: 31/255, green: 41/255, blue: 55/255, alpha: 1.0)    // #1f2937
    })

    static let textSecondary = Color(UIColor { traitCollection in
        traitCollection.userInterfaceStyle == .dark
            ? UIColor(red: 160/255, green: 160/255, blue: 176/255, alpha: 1.0) // #a0a0b0
            : UIColor(red: 107/255, green: 114/255, blue: 128/255, alpha: 1.0) // #6b7280
    })

    func withOpacity(_ opacity: Double) -> Color {
        return self.opacity(opacity)
    }
}
