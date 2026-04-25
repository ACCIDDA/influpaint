#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# explore_seasons.R
#
# Interactive Shiny app for exploring RSV_FILLED.parquet and picking
# held-out seasons for validation_split.R.
#
# Run from the project root:
#   shiny::runApp("RSV/workflow/explore_seasons.R")
# ---------------------------------------------------------------------------

library(shiny)
library(arrow)
library(dplyr)
library(ggplot2)
library(scales)
library(here)

rsv_full <- read_parquet(here("RSV/data/RSV_FILLED.parquet"))

all_seasons   <- sort(unique(rsv_full$fluseason))
all_h1        <- sort(unique(rsv_full$datasetH1))
all_locations <- sort(unique(rsv_full$location_code))

# ---- UI -------------------------------------------------------------------

ui <- fluidPage(
  titlePanel("RSV Season Explorer"),

  sidebarLayout(
    sidebarPanel(
      width = 3,

      h4("Held-out seasons"),
      p("Check seasons to hold out for validation.",
        style = "font-size:0.9em; color:#555"),
      checkboxGroupInput("held_out", NULL,
        choices  = all_seasons,
        selected = NULL
      ),

      hr(),
      h4("Paste into validation_split.R"),
      verbatimTextOutput("snippet"),

      hr(),
      h4("Display filters"),
      checkboxGroupInput("h1_sel", "Data sources",
        choices  = all_h1,
        selected = all_h1
      ),
      radioButtons("scale", "Colour scale",
        choices  = c("Log10" = "log10", "Linear" = "linear"),
        selected = "log10",
        inline   = TRUE
      )
    ),

    mainPanel(
      width = 9,
      tabsetPanel(

        # -- Tab 1: heatmap ---------------------------------------------------
        tabPanel(
          "Heatmap",
          br(),
          p("Median value per (location, week) cell. Held-out seasons are outlined in red."),
          plotOutput("heatmap", height = "680px")
        ),

        # -- Tab 2: coverage --------------------------------------------------
        tabPanel(
          "Season coverage",
          br(),
          p("Row counts per data source and season. Only selected data sources shown.
            Held-out seasons are shaded."),
          plotOutput("coverage", height = "480px")
        ),

        # -- Tab 3: missingness -----------------------------------------------
        tabPanel(
          "Missingness",
          br(),
          fluidRow(
            column(4, selectInput("miss_h1", "Data source",
              choices  = all_h1,
              selected = all_h1[1]
            )),
            column(4, radioButtons("miss_by", "Break down by",
              choices  = c("Location" = "location", "Week" = "week"),
              selected = "location",
              inline   = TRUE
            ))
          ),
          plotOutput("missingness", height = "500px")
        ),

        # -- Tab 4: time series -----------------------------------------------
        tabPanel(
          "Time series",
          br(),
          fluidRow(
            column(4, selectInput("ts_location", "Location",
              choices  = c("All locations (median)" = "_all", all_locations),
              selected = "_all"
            )),
            column(4, selectInput("ts_h1", "Data source",
              choices  = all_h1,
              selected = all_h1[1]
            ))
          ),
          plotOutput("timeseries", height = "430px")
        )
      )
    )
  )
)

# ---- Server ---------------------------------------------------------------

server <- function(input, output, session) {

  held <- reactive(as.integer(input$held_out))

  # data filtered to selected H1 sources, median across samples
  rsv_median <- reactive({
    rsv_full |>
      filter(datasetH1 %in% input$h1_sel) |>
      group_by(location_code, fluseason, fluseason_week) |>
      summarise(value = median(value, na.rm = TRUE), .groups = "drop")
  })

  # ---- Snippet --------------------------------------------------------------
  output$snippet <- renderText({
    h <- held()
    if (length(h) == 0)
      'VALIDATION_SEASONS <- c()'
    else
      paste0("VALIDATION_SEASONS <- c(", paste(h, collapse = ", "), ")")
  })

  # ---- Heatmap --------------------------------------------------------------
  output$heatmap <- renderPlot({
    df <- rsv_median()

    p <- ggplot(df, aes(x = fluseason_week, y = location_code, fill = value)) +
      geom_tile() +
      facet_wrap(~fluseason, nrow = 1) +
      labs(x = "Week within season", y = NULL, fill = "Median value") +
      theme_bw(base_size = 10) +
      theme(
        strip.background = element_rect(fill = "grey92"),
        strip.text       = element_text(face = "bold"),
        axis.text.y      = element_text(size = 6),
        axis.text.x      = element_text(size = 7),
        panel.spacing    = unit(0.1, "lines"),
        legend.position  = "bottom",
        legend.key.width = unit(2, "cm")
      )

    if (input$scale == "log10") {
      p <- p + scale_fill_viridis_c(trans = "log10", na.value = "grey90",
                                    labels = label_comma())
    } else {
      p <- p + scale_fill_viridis_c(na.value = "grey90", labels = label_comma())
    }

    # Red outline on held-out season panels
    if (length(held()) > 0) {
      p <- p + geom_rect(
        data        = data.frame(fluseason = held()),
        aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf),
        fill        = NA, colour = "red", linewidth = 1.3,
        inherit.aes = FALSE
      )
    }
    p
  })

  # ---- Coverage -------------------------------------------------------------
  output$coverage <- renderPlot({
    counts <- rsv_full |>
      filter(datasetH1 %in% input$h1_sel) |>
      count(datasetH1, fluseason)

    p <- ggplot(counts, aes(x = factor(fluseason), y = n, fill = datasetH1)) +
      geom_col(position = "dodge") +
      scale_y_continuous(labels = label_comma()) +
      scale_fill_brewer(palette = "Set2") +
      labs(x = "Season", y = "Rows", fill = NULL,
           title = "Row counts per data source × season") +
      theme_bw(base_size = 12) +
      theme(axis.text.x = element_text(angle = 30, hjust = 1),
            legend.position = "top")

    # Shade held-out seasons
    if (length(held()) > 0) {
      p <- p + geom_rect(
        data        = data.frame(fluseason = factor(held())),
        aes(xmin = as.numeric(fluseason) - 0.5,
            xmax = as.numeric(fluseason) + 0.5,
            ymin = -Inf, ymax = Inf),
        fill        = "red", alpha = 0.12,
        inherit.aes = FALSE
      )
    }
    p
  })

  # ---- Missingness ----------------------------------------------------------
  output$missingness <- renderPlot({
    df <- rsv_full |> filter(datasetH1 == input$miss_h1)

    # Full grid: every season × location × week that *could* exist
    grid <- expand.grid(
      fluseason      = all_seasons,
      location_code  = all_locations,
      fluseason_week = seq(min(df$fluseason_week), max(df$fluseason_week)),
      stringsAsFactors = FALSE
    )

    present <- df |>
      filter(!is.na(value)) |>
      distinct(fluseason, location_code, fluseason_week)

    if (input$miss_by == "location") {
      pct <- grid |>
        left_join(present |> mutate(.present = TRUE),
                  by = c("fluseason", "location_code", "fluseason_week")) |>
        group_by(fluseason, location_code) |>
        summarise(pct_missing = mean(is.na(.present)) * 100, .groups = "drop")

      p <- ggplot(pct, aes(x = factor(fluseason), y = location_code,
                           fill = pct_missing)) +
        geom_tile() +
        scale_fill_gradient(low = "white", high = "#d73027",
                            limits = c(0, 100), labels = label_percent(scale = 1)) +
        labs(x = "Season", y = NULL, fill = "% weeks\nmissing",
             title = paste0(input$miss_h1, " — % missing weeks per location × season")) +
        theme_bw(base_size = 11) +
        theme(axis.text.y = element_text(size = 7),
              legend.key.height = unit(1.5, "cm"))

    } else {
      pct <- grid |>
        left_join(present |> mutate(.present = TRUE),
                  by = c("fluseason", "location_code", "fluseason_week")) |>
        group_by(fluseason, fluseason_week) |>
        summarise(pct_missing = mean(is.na(.present)) * 100, .groups = "drop")

      p <- ggplot(pct, aes(x = fluseason_week, y = factor(fluseason),
                           fill = pct_missing)) +
        geom_tile() +
        scale_fill_gradient(low = "white", high = "#d73027",
                            limits = c(0, 100), labels = label_percent(scale = 1)) +
        labs(x = "Week within season", y = "Season", fill = "% locations\nmissing",
             title = paste0(input$miss_h1, " — % missing locations per week × season")) +
        theme_bw(base_size = 11) +
        theme(legend.key.height = unit(1.5, "cm"))
    }

    # Red outline on held-out seasons
    if (length(held()) > 0) {
      if (input$miss_by == "location") {
        p <- p + geom_rect(
          data        = data.frame(fluseason = factor(held())),
          aes(xmin = as.numeric(fluseason) - 0.5,
              xmax = as.numeric(fluseason) + 0.5,
              ymin = -Inf, ymax = Inf),
          fill = NA, colour = "red", linewidth = 1.2, inherit.aes = FALSE
        )
      } else {
        p <- p + geom_rect(
          data        = data.frame(fluseason = factor(held())),
          aes(xmin = -Inf, xmax = Inf,
              ymin = as.numeric(fluseason) - 0.5,
              ymax = as.numeric(fluseason) + 0.5),
          fill = NA, colour = "red", linewidth = 1.2, inherit.aes = FALSE
        )
      }
    }
    p
  })

  # ---- Time series ----------------------------------------------------------
  output$timeseries <- renderPlot({
    df <- rsv_full |> filter(datasetH1 == input$ts_h1)

    if (input$ts_location != "_all")
      df <- df |> filter(location_code == input$ts_location)

    df <- df |>
      mutate(abs_week = (fluseason - min(fluseason)) * 53L + fluseason_week) |>
      group_by(abs_week, fluseason, fluseason_week) |>
      summarise(
        med = median(value, na.rm = TRUE),
        q10 = quantile(value, 0.10, na.rm = TRUE),
        q90 = quantile(value, 0.90, na.rm = TRUE),
        .groups = "drop"
      )

    season_labels <- df |>
      group_by(fluseason) |>
      summarise(mid = median(abs_week), .groups = "drop")

    p <- ggplot(df, aes(x = abs_week)) +
      geom_ribbon(aes(ymin = q10, ymax = q90), fill = "steelblue", alpha = 0.25) +
      geom_line(aes(y = med), colour = "steelblue", linewidth = 0.7) +
      scale_x_continuous(breaks = season_labels$mid,
                         labels = season_labels$fluseason) +
      labs(x = NULL, y = "Value",
           title = paste0(input$ts_h1, " — ",
             if (input$ts_location == "_all") "all locations (median)" else input$ts_location)) +
      theme_bw(base_size = 12)

    # Shade held-out seasons
    if (length(held()) > 0) {
      shade <- df |>
        filter(fluseason %in% held()) |>
        group_by(fluseason) |>
        summarise(xmin = min(abs_week) - 0.5, xmax = max(abs_week) + 0.5,
                  .groups = "drop")
      p <- p + geom_rect(
        data = shade,
        aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
        fill = "red", alpha = 0.10, inherit.aes = FALSE
      )
    }
    p
  })
}

shinyApp(ui, server)
