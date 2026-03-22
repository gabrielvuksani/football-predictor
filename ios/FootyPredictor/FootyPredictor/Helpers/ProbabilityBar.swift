import SwiftUI

struct ProbabilityBar: View {
    let homeProb: Double
    let drawProb: Double
    let awayProb: Double
    let homeTeam: String
    let awayTeam: String

    private var total: Double {
        let sum = homeProb + drawProb + awayProb
        return sum > 0 ? sum : 1.0
    }

    var body: some View {
        VStack(spacing: 8) {
            GeometryReader { geometry in
                let totalWidth = geometry.size.width
                let spacing: CGFloat = 4
                let availableWidth = totalWidth - (spacing * 2)
                let homeWidth = max(availableWidth * CGFloat(homeProb / total), 20)
                let drawWidth = max(availableWidth * CGFloat(drawProb / total), 20)
                let awayWidth = max(availableWidth * CGFloat(awayProb / total), 20)
                let scaledTotal = homeWidth + drawWidth + awayWidth
                let scale = availableWidth / scaledTotal

                HStack(spacing: spacing) {
                    BarSegment(value: homeProb, color: .accentBlue, label: String(format: "%.0f%%", homeProb * 100))
                        .frame(width: homeWidth * scale)
                    BarSegment(value: drawProb, color: .warningYellow, label: String(format: "%.0f%%", drawProb * 100))
                        .frame(width: drawWidth * scale)
                    BarSegment(value: awayProb, color: .dangerRed, label: String(format: "%.0f%%", awayProb * 100))
                        .frame(width: awayWidth * scale)
                }
            }
            .frame(height: 28)

            HStack {
                Text(homeTeam)
                    .font(.caption)
                    .lineLimit(1)
                Spacer()
                Text("Draw")
                    .font(.caption2)
                Spacer()
                Text(awayTeam)
                    .font(.caption)
                    .lineLimit(1)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(homeTeam) \(String(format: "%.0f", homeProb * 100)) percent, Draw \(String(format: "%.0f", drawProb * 100)) percent, \(awayTeam) \(String(format: "%.0f", awayProb * 100)) percent")
    }
}

struct BarSegment: View {
    let value: Double
    let color: Color
    let label: String

    @Environment(\.colorScheme) private var colorScheme

    private var textColor: Color {
        if value > 0.15 {
            return .white
        }
        return colorScheme == .dark ? .white : .primary
    }

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 4)
                .fill(color)

            Text(label)
                .font(.system(size: 10, weight: .semibold))
                .foregroundColor(textColor)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .accessibilityLabel("\(label) probability")
    }
}

#Preview {
    ProbabilityBar(homeProb: 0.45, drawProb: 0.25, awayProb: 0.30, homeTeam: "Man City", awayTeam: "Chelsea")
        .padding()
}
