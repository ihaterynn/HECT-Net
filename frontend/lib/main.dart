import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:provider/provider.dart';
import 'dart:io' show File, Directory, Platform;
import 'package:path/path.dart' as path;
import 'package:flutter/foundation.dart' show kIsWeb;

import 'package:snackly/screens/auth_screen.dart';
import 'package:snackly/screens/meal_timeline_screen.dart';
import 'package:snackly/screens/scan_screen.dart';
import 'package:snackly/components/bottom_navbar.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  String? supabaseUrl;
  String? supabaseAnonKey;
  User? initialUser; // To store the initially fetched user

  if (kIsWeb) {
    print("DEBUG: Web platform detected. Using hardcoded Supabase credentials.");
    supabaseUrl = 'https://ckvrqahqiynyioiapbld.supabase.co';
    supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNrdnJxYWhxaXlueWlvaWFwYmxkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NzAyODIsImV4cCI6MjA2MzQ0NjI4Mn0.GosVjKMI4CRKa84sm9qIumzDiEhdsuz5cS_ToU9eRYo';
  } else {
    print("DEBUG: Non-web platform detected. Attempting to load .env file.");
    final String envPath = '../.env';
    try {
      await dotenv.load(fileName: envPath);
      supabaseUrl = dotenv.env['SUPABASE_URL'];
      supabaseAnonKey = dotenv.env['SUPABASE_ANON_KEY'];
      print("DEBUG: Loaded SUPABASE_URL from env: $supabaseUrl");
      print("DEBUG: Loaded SUPABASE_ANON_KEY from env (first 10 chars): ${supabaseAnonKey?.substring(0, 10)}...");
    } catch (e) {
      print("ERROR loading .env file at $envPath: $e");
      runApp(ErrorApp(message: "Failed to load .env file from $envPath. Error: $e"));
      return;
    }
  }

  if (supabaseUrl == null || supabaseUrl.isEmpty || supabaseAnonKey == null || supabaseAnonKey.isEmpty) {
    print("ERROR: Supabase URL or Anon Key is missing or empty.");
    String errorMessage = "Supabase URL or Anon Key is missing. ";
    if (kIsWeb) {
      errorMessage += "Check hardcoded values in main.dart.";
    } else {
      errorMessage += "Check your .env file at ../.env.";
    }
    runApp(ErrorApp(message: errorMessage));
    return;
  }

  // Initialize Supabase first
  await Supabase.initialize(
    url: supabaseUrl,
    anonKey: supabaseAnonKey,
  );
  print("Supabase has been initialized (main.dart)");

  // Get the initial user state *after* initialization
  initialUser = Supabase.instance.client.auth.currentUser;
  if(initialUser == null) {
    print("DEBUG: No initial user session found.");
  } else {
    print("DEBUG: Initial user session FOUND for: ${initialUser.email}");
  }

  runApp(MyApp(initialUser: initialUser)); // Pass initial user to MyApp
}

// ErrorApp remains the same
class ErrorApp extends StatelessWidget {
  final String message;
  const ErrorApp({Key? key, required this.message}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(title: const Text('Configuration Error')),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              message,
              style: const TextStyle(color: Colors.red, fontSize: 16),
              textAlign: TextAlign.center,
            ),
          ),
        ),
      ),
    );
  }
}


class MyApp extends StatelessWidget {
  final User? initialUser; // Accept initial user
  const MyApp({Key? key, this.initialUser}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Snackly',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        inputDecorationTheme: InputDecorationTheme(
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(8),
          ),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.blueAccent,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 20),
          ),
        ),
      ),
      home: AuthStateChangeHandler(initialUser: initialUser), // Pass to AuthStateChangeHandler
    );
  }
}

class AuthStateChangeHandler extends StatelessWidget {
  final User? initialUser; // Accept initial user
  const AuthStateChangeHandler({Key? key, this.initialUser}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Use the initialUser for the first build if available, then rely on the stream
    final initialSession = initialUser?.id != null;
    print("DEBUG AuthStateChangeHandler: Initial session active based on initialUser: $initialSession");

    return StreamBuilder<AuthState>(
      // initialData can be tricky with Supabase's stream, so we handle initial state above
      stream: Supabase.instance.client.auth.onAuthStateChange,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting && !initialSession && snapshot.data == null) {
          print("DEBUG AuthStateChangeHandler: Waiting for auth stream (no initial session, no stream data yet)");
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }

        // Determine current session: prioritize stream data, then initialUser, then null
        // Refined logic for initialUser: only use it if stream is not yet active with data
        final Session? session = snapshot.data?.session ?? 
                               (initialUser?.id != null && snapshot.connectionState != ConnectionState.active && snapshot.data == null 
                                 ? Session(accessToken: 'dummy', user: initialUser!, tokenType: 'bearer') 
                                 : null);
        final bool hasSession = session != null;
        
        print("DEBUG AuthStateChangeHandler: Event: ${snapshot.data?.event}, Has Session: $hasSession, Current User from Supabase: ${Supabase.instance.client.auth.currentUser?.email}");

        if (hasSession) {
          return const BottomNavbarWidget();
        } else {
          print("DEBUG AuthStateChangeHandler: No session, navigating to AuthScreen.");
          return const AuthScreen();
        }
      },
    );
  }
}
