import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(const EdgeAiApp());
}

class EdgeAiApp extends StatelessWidget {
  const EdgeAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: BridgeTestPage(),
    );
  }
}

class BridgeTestPage extends StatefulWidget {
  const BridgeTestPage({super.key});

  @override
  State<BridgeTestPage> createState() => _BridgeTestPageState();
}

class _BridgeTestPageState extends State<BridgeTestPage> {
  static const platform = MethodChannel('edge_ai/native');

  String result = 'Bridge not tested';

  Future<void> testBridge() async {
    try {
      final response = await platform.invokeMethod<String>('ping');

      setState(() {
        result = response ?? 'No response';
      });
    } on PlatformException catch (e) {
      setState(() {
        result = 'Error: ${e.message}';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Edge AI Bank Transfers'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(result),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: testBridge,
              child: const Text('Test native bridge'),
            ),
          ],
        ),
      ),
    );
  }
}