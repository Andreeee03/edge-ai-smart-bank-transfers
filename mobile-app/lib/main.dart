import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(const EdgeAiApp());
}

class EdgeAiApp extends StatelessWidget {
  const EdgeAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Edge AI Smart Bank Transfers',
      home: const BridgeTestPage(),
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

  String _status = 'Ready';
  bool _busy = false;

  Future<void> _testBridge() async {
    setState(() {
      _status = 'Testing native bridge...';
    });

    try {
      final result = await platform.invokeMethod<String>('ping');

      setState(() {
        _status = result ?? 'No response';
      });
    } on PlatformException catch (e) {
      setState(() {
        _status = 'Error: ${e.message}';
      });
    }
  }

  Future<void> _loadModel() async {
    setState(() {
      _busy = true;
      _status = 'Loading Q5 model...';
    });

    try {
      final result = await platform.invokeMethod<String>('loadModel');

      setState(() {
        _status = result ?? 'No response';
      });
    } on PlatformException catch (e) {
      setState(() {
        _status = 'Error: ${e.message}';
      });
    } finally {
      setState(() {
        _busy = false;
      });
    }
  }

  Future<void> _createContext() async {
    setState(() {
      _busy = true;
      _status = 'Creating llama context...';
    });

    try {
      final result =
          await platform.invokeMethod<String>('createContext');

      setState(() {
        _status = result ?? 'No response';
      });
    } on PlatformException catch (e) {
      setState(() {
        _status = 'Error: ${e.message}';
      });
    } finally {
      setState(() {
        _busy = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Edge AI Smart Bank Transfers'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton(
              onPressed: _busy ? null : _testBridge,
              child: const Text('Test native bridge'),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _busy ? null : _loadModel,
              child: const Text('Load Q5 model'),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _busy ? null : _createContext,
              child: const Text('Create context'),
            ),
            const SizedBox(height: 24),
            const Text(
              'Status:',
              style: TextStyle(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            SelectableText(_status),
          ],
        ),
      ),
    );
  }
}
